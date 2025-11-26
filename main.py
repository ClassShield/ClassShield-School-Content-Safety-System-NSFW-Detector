import os
import io
import time
import hashlib
import json
import logging
import base64
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
from functools import wraps
from PIL import Image, ImageDraw, ImageFilter
import requests
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

try:
    from nudenet import NudeClassifier
    LOCAL_MODEL_AVAILABLE = True
    classifier = NudeClassifier()
except Exception as e:
    LOCAL_MODEL_AVAILABLE = False
    classifier = None
    print("Local NudeNet model not available:", e)

SIGHT_USER = os.getenv("SIGHTENGINE_API_USER", "")
SIGHT_SECRET = os.getenv("SIGHTENGINE_API_SECRET", "")

app = Flask(__name__)
app.secret_key = os.getenv("SESSION_SECRET", "dev-secret-key-change-in-production")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

flagged_items = []
audit_log = []

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nsfw_audit.log'),
        logging.StreamHandler()
    ]
)
nsfw_logger = logging.getLogger('nsfw_audit')

def require_admin(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_authenticated'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

def compute_image_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()

def log_audit(action, details):
    audit_log.append({
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details
    })
    if len(audit_log) > 1000:
        audit_log.pop(0)

def check_with_sightengine_bytes(image_bytes):
    if not SIGHT_USER or not SIGHT_SECRET:
        return None
    url = "https://api.sightengine.com/1.0/check.json"
    files = {'media': ('image.jpg', image_bytes)}
    data = {
        'models': 'nudity-2.0,wad',
        'api_user': SIGHT_USER,
        'api_secret': SIGHT_SECRET
    }
    try:
        resp = requests.post(url, files=files, data=data, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print("Sightengine error:", e)
        return None

def local_nudenet_classify_bytes(image_bytes):
    if not LOCAL_MODEL_AVAILABLE:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        res = classifier.classify(img)
        return res
    except Exception as e:
        print("Local classifier error:", e)
        return None

def calculate_skin_ratio(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        arr = np.array(img)
        h, w = arr.shape[0], arr.shape[1]
        total_pixels = h * w

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        skin_mask = (
            (r > 95) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            (abs(r - g) > 15) &
            (r - g > 0)
        )

        skin_pixels = np.sum(skin_mask)
        skin_ratio = (skin_pixels / total_pixels) * 100

        return round(skin_ratio, 2)
    except Exception as e:
        print("Skin ratio calculation error:", e)
        return 0

def generate_risk_heatmap(image_bytes, skin_ratio, decision):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        arr = np.array(img)
        h, w = arr.shape[0], arr.shape[1]

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        skin_mask = (
            (r > 95) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            (abs(r - g) > 15) &
            (r - g > 0)
        )

        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=15))
        darkened_arr = (np.array(blurred_img) * 0.3).astype(np.uint8)
        darkened_img = Image.fromarray(darkened_arr)

        overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        overlay_arr = np.array(overlay)

        non_skin_mask = ~skin_mask
        overlay_arr[non_skin_mask] = [40, 40, 40, 245]

        if decision == "BLOCK":
            overlay_arr[skin_mask] = [220, 20, 20, 250]
        elif decision == "REVIEW":
            overlay_arr[skin_mask] = [220, 200, 20, 248]
        else:
            overlay_arr[skin_mask] = [20, 200, 20, 246]

        overlay_img = Image.fromarray(overlay_arr, mode='RGBA')

        target_width = 400
        target_height = int(400 * h / w)
        resized_darkened = darkened_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        resized_overlay = overlay_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        composite = Image.new('RGBA', (target_width, target_height))
        composite.paste(resized_darkened.convert('RGBA'))
        composite = Image.alpha_composite(composite, resized_overlay)

        buffered = io.BytesIO()
        composite.convert('RGB').save(buffered, format="JPEG", quality=85)
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return f"data:image/jpeg;base64,{heatmap_base64}"
    except Exception as e:
        print("Heatmap generation error:", e)
        nsfw_logger.error(f"Heatmap generation failed: {e}")
        return None

def detect_contextual_keywords(sightengine_response):
    keywords_detected = {
        'bra_lingerie': False,
        'bed_context': False,
        'swimwear': False,
        'beach_context': False
    }

    if not sightengine_response:
        return keywords_detected

    try:
        nudity = sightengine_response.get('nudity', {})

        if 'lingerie' in str(nudity).lower() or 'bra' in str(nudity).lower():
            keywords_detected['bra_lingerie'] = True

        context = sightengine_response.get('weapon', {})
        raw_categories = sightengine_response.get('nudity', {}).get('raw_categories', {})

        if 'bed' in str(sightengine_response).lower():
            keywords_detected['bed_context'] = True

        if 'swimwear' in str(nudity).lower() or 'bikini' in str(nudity).lower():
            keywords_detected['swimwear'] = True

        if 'beach' in str(sightengine_response).lower() or 'pool' in str(sightengine_response).lower():
            keywords_detected['beach_context'] = True

    except Exception as e:
        print("Keyword detection error:", e)

    return keywords_detected

def determine_final_decision(score, skin_ratio, keywords, evidence):
    decision = "SAFE"
    reason = "No inappropriate content detected"
    action = "allow"

    if score <= 0.15:
        decision = "SAFE"
        reason = "Score below SAFE threshold (≤0.15)"
        action = "allow"
    elif 0.15 < score < 0.35:
        decision = "REVIEW"
        reason = "Score in REVIEW range (0.15-0.35)"
        action = "send_to_admin_review"
    elif score >= 0.35:
        decision = "BLOCK"
        reason = "Score above BLOCK threshold (≥0.35)"
        action = "block_and_send_to_admin_review"

    if keywords.get('bra_lingerie'):
        if decision == "SAFE":
            decision = "REVIEW"
            reason += " | Bra/lingerie detected"
            action = "send_to_admin_review"
        elif decision in ["REVIEW", "BLOCK"]:
            reason += " | Bra/lingerie detected"

    if keywords.get('bed_context') and skin_ratio > 40:
        if decision == "SAFE":
            decision = "REVIEW"
            reason += " | Bed context + high skin ratio (>40%)"
            action = "send_to_admin_review"
        elif decision in ["REVIEW", "BLOCK"]:
            reason += " | Bed context + high skin ratio (>40%)"

    if keywords.get('swimwear') and keywords.get('beach_context'):
        if decision == "BLOCK":
            decision = "REVIEW"
            reason += " | Swimwear + beach context detected (lower risk)"
            action = "send_to_admin_review"
        elif decision in ["SAFE", "REVIEW"]:
            reason += " | Swimwear + beach context detected"

    if skin_ratio > 60:
        if decision == "SAFE":
            decision = "REVIEW"
            reason += f" | Suspicious skin ratio ({skin_ratio}% > 60%)"
            action = "send_to_admin_review"
        elif decision in ["REVIEW", "BLOCK"]:
            reason += f" | Suspicious skin ratio ({skin_ratio}% > 60%)"
    elif skin_ratio > 40:
        if decision == "SAFE":
            decision = "REVIEW"
            reason += f" | High skin ratio ({skin_ratio}% > 40%)"
            action = "send_to_admin_review"
        elif decision in ["REVIEW", "BLOCK"]:
            reason += f" | High skin ratio ({skin_ratio}% > 40%)"

    return decision, reason, action

@app.route("/")
def index():
    return render_template("index.html", local_model=LOCAL_MODEL_AVAILABLE)

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == ADMIN_PASSWORD:
            session['admin_authenticated'] = True
            log_audit("admin_login", {"success": True})
            return redirect(url_for('admin_dashboard'))
        else:
            log_audit("admin_login_failed", {"success": False})
            return render_template("admin_login.html", error="Invalid password")
    return render_template("admin_login.html")

@app.route("/admin/logout")
def admin_logout():
    session.pop('admin_authenticated', None)
    log_audit("admin_logout", {})
    return redirect(url_for('index'))

@app.route("/admin")
@require_admin
def admin_dashboard():
    return render_template("admin.html", flagged_count=len(flagged_items))

@app.route("/api/flagged")
@require_admin
def get_flagged():
    return jsonify(flagged_items)

@app.route("/api/audit")
@require_admin
def get_audit_log():
    return jsonify(audit_log[-100:])

@app.route("/api/review/<int:item_id>/<action>", methods=["POST"])
@require_admin
def review_action(item_id, action):
    if item_id < len(flagged_items):
        item = flagged_items[item_id]
        item['review_status'] = action
        item['reviewed_at'] = datetime.now().isoformat()
        log_audit(f"review_{action}", {"item_id": item_id, "image_hash": item['image_hash']})
        nsfw_logger.info(f"REVIEW ACTION | Item #{item_id} | Hash: {item['image_hash'][:16]} | Action: {action}")
        return jsonify({"success": True, "message": f"Item {action}ed"})
    return jsonify({"success": False, "message": "Item not found"}), 404

@app.route("/api/revoke/<int:item_id>", methods=["POST"])
@require_admin
def revoke_review(item_id):
    if item_id < len(flagged_items):
        item = flagged_items[item_id]
        previous_status = item.get('review_status', 'unknown')
        item['review_status'] = 'pending'
        item['reviewed_at'] = None
        item['revoked_at'] = datetime.now().isoformat()
        log_audit("review_revoked", {
            "item_id": item_id,
            "image_hash": item['image_hash'],
            "previous_status": previous_status
        })
        nsfw_logger.warning(f"REVIEW REVOKED | Item #{item_id} | Hash: {item['image_hash'][:16]} | Previous: {previous_status}")
        return jsonify({"success": True, "message": "Review revoked, item returned to pending queue"})
    return jsonify({"success": False, "message": "Item not found"}), 404

@app.route("/ethical-ai")
def ethical_ai():
    return render_template("ethical_ai.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/bias-testing")
def bias_testing():
    return render_template("bias_testing.html")

@app.route("/education")
def education():
    return render_template("education.html")

@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")

@app.route("/deployment")
def deployment():
    return render_template("deployment.html")

@app.route("/demo-plan")
def demo_plan():
    return render_template("demo_plan.html")

@app.route("/submission")
def submission():
    return render_template("submission.html")

@app.route("/screenshot")
def screenshot_page():
    return render_template("screenshot.html")

@app.route("/scan", methods=["POST"])
def scan_image():
    if 'image' not in request.files:
        return jsonify({"error": "no image file"}), 400

    f = request.files['image']
    content = f.read()
    image_hash = compute_image_hash(content)

    result = {
        "timestamp": int(time.time()),
        "image_hash": image_hash,
        "methods": [],
        "decision": "SAFE",
        "reason": "",
        "evidence": {},
        "thresholds_used": {
            "safe_threshold": 0.15,
            "review_threshold_lower": 0.15,
            "review_threshold_upper": 0.35,
            "block_threshold": 0.35,
            "skin_ratio_review": 40,
            "skin_ratio_suspicious": 60
        },
        "keywords_detected": {},
        "skin_ratio": 0
    }

    skin_ratio = calculate_skin_ratio(content)
    result['skin_ratio'] = skin_ratio
    result['methods'].append("skin_ratio_detection")

    highest_score = 0.0
    sightengine_response = None

    local = local_nudenet_classify_bytes(content)
    if local:
        try:
            first = next(iter(local.values()))
            unsafe_score = float(first.get("unsafe", 0))
            safe_score = float(first.get("safe", 0))
            result['methods'].append("local_nudenet")
            result['evidence']['nudenet'] = {
                "unsafe": round(unsafe_score, 4),
                "safe": round(safe_score, 4),
                "confidence": round(max(unsafe_score, safe_score), 4)
            }
            highest_score = max(highest_score, unsafe_score)
        except Exception as e:
            print("Parsing local output error:", e)
            nsfw_logger.error(f"NudeNet parsing error for {image_hash}: {e}")

    sight = check_with_sightengine_bytes(content)
    if sight:
        sightengine_response = sight
        result['methods'].append("sightengine")
        nudity = sight.get('nudity', {})
        sexual_activity = nudity.get('sexual_activity', 0)
        sexual_display = nudity.get('sexual_display', 0)
        combined_score = sexual_activity + sexual_display

        result['evidence']['sightengine'] = {
            "sexual_activity": round(sexual_activity, 4),
            "sexual_display": round(sexual_display, 4),
            "combined_score": round(combined_score, 4),
            "raw_response": nudity
        }

        highest_score = max(highest_score, combined_score)

    keywords = detect_contextual_keywords(sightengine_response)
    result['keywords_detected'] = keywords

    decision, reason, action = determine_final_decision(
        highest_score,
        skin_ratio,
        keywords,
        result['evidence']
    )

    result['decision'] = decision
    result['reason'] = reason
    result['action'] = action
    result['primary_score'] = round(highest_score, 4)
    result['summary'] = f"Decision: {decision}. Action: {action.replace('_', ' ').title()}. Score: {highest_score:.4f}, Skin Ratio: {skin_ratio}%"

    nsfw_logger.info(
        f"SCAN | Hash: {image_hash[:16]} | Decision: {decision} | Score: {highest_score:.4f} | "
        f"Skin: {skin_ratio}% | Keywords: {keywords} | Reason: {reason}"
    )

    if decision in ["REVIEW", "BLOCK"]:
        heatmap_data = generate_risk_heatmap(content, skin_ratio, decision)

        flagged_item = {
            "id": len(flagged_items),
            "image_hash": image_hash,
            "decision": decision,
            "reason": reason,
            "evidence": result['evidence'],
            "methods": result['methods'],
            "skin_ratio": skin_ratio,
            "keywords_detected": keywords,
            "primary_score": round(highest_score, 4),
            "flagged_at": datetime.now().isoformat(),
            "review_status": "pending",
            "heatmap": heatmap_data
        }
        flagged_items.append(flagged_item)
        log_audit("image_flagged", {
            "image_hash": image_hash,
            "decision": decision,
            "methods": result['methods'],
            "skin_ratio": skin_ratio,
            "keywords": keywords
        })
        nsfw_logger.warning(
            f"FLAGGED | Hash: {image_hash[:16]} | Decision: {decision} | "
            f"Score: {highest_score:.4f} | Skin: {skin_ratio}%"
        )

    log_audit("image_scanned", {
        "image_hash": image_hash,
        "decision": decision,
        "methods": result['methods']
    })

    return jsonify(result)

@app.route("/api/status")
def status():
    return jsonify({
        "project": "ClassShield School Safety - NSFW Detector (Prototype) - Powered By AnveshAI",
        "status": "ok",
        "local_model": LOCAL_MODEL_AVAILABLE,
        "sightengine_configured": bool(SIGHT_USER and SIGHT_SECRET),
        "flagged_items": len(flagged_items),
        "total_scans": len([log for log in audit_log if log['action'] == 'image_scanned'])
    })

@app.route("/api/screenshot", methods=["POST"])
def take_screenshot():
    data = request.get_json()
    url = data.get("url", request.host_url)

    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(2)

        screenshot_bytes = driver.get_screenshot_as_png()
        driver.quit()

        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        log_audit("screenshot_taken", {"url": url, "timestamp": datetime.now().isoformat()})
        nsfw_logger.info(f"SCREENSHOT | URL: {url}")

        return jsonify({
            "success": True,
            "screenshot": f"data:image/png;base64,{screenshot_base64}",
            "url": url,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        nsfw_logger.error(f"Screenshot error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)