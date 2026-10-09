
import os
import json
import hashlib
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Weed Management",
    page_icon="🌱",
    layout="wide"
)

# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "improved_weed_unet_final.keras"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR, "best_threshold.json"
)

# =========================================================
# LOAD MODEL AND THRESHOLD
# =========================================================

@st.cache_resource
def load_weed_model():
    return tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )


@st.cache_data
def load_threshold():
    with open(THRESHOLD_PATH, "r") as file:
        data = json.load(file)

    return float(data["best_threshold"])


try:
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found. Place "
            "'improved_weed_unet_final.keras' "
            "beside app.py."
        )
        st.stop()

    if not os.path.exists(THRESHOLD_PATH):
        st.error(
            "Threshold file not found. Place "
            "'best_threshold.json' beside app.py."
        )
        st.stop()

    model = load_weed_model()
    best_threshold = load_threshold()

except Exception as error:
    st.error("Unable to load the model or threshold.")
    st.exception(error)
    st.stop()

# =========================================================
# SESSION STATE
# =========================================================

for key, default in [
    ("single_result", None),
    ("single_result_key", None),
    ("comparison_result", None),
    ("comparison_result_key", None),
    ("previous_mode", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #f5f8f6;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #14532d;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 20px;
    }

    .section-heading {
        font-size: 25px;
        font-weight: 700;
        color: #14532d;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    [data-testid="stSidebar"] {
        background-color: #eef6f0;
    }

    div[role="radiogroup"] {
        gap: 20px;
    }

    div[role="radiogroup"] label {
        background-color: white;
        border: 2px solid #dce5df;
        border-radius: 12px;
        padding: 16px 24px;
        min-height: 55px;
        font-size: 17px;
        font-weight: 600;
    }

    div[role="radiogroup"] label:has(input:checked) {
        background-color: #e1f3e5;
        border-color: #15803d;
    }

    div.stButton > button {
        min-height: 48px;
        padding: 10px 24px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# IMAGE ANALYSIS
# =========================================================

def analyze_image(image):
    """Predict a binary foreground mask and its pixel coverage."""

    image = image.convert("RGB")

    resized_image = image.resize(
        (256, 256),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    ) / 255.0

    image_batch = np.expand_dims(image_array, axis=0)

    prediction = model.predict(
        image_batch,
        verbose=0
    )

    probability_map = prediction[0, :, :, 0]

    binary_mask = (
        probability_map >= best_threshold
    ).astype(np.uint8)

    coverage = float(np.mean(binary_mask) * 100.0)

    mask_image = Image.fromarray(
        binary_mask * 255
    ).resize(
        image.size,
        Image.Resampling.NEAREST
    )

    original_array = np.asarray(
        image,
        dtype=np.uint8
    ).copy()

    full_size_mask = np.asarray(mask_image) > 0

    overlay_array = original_array.copy()

    red = np.zeros_like(original_array)
    red[:, :, 0] = 255

    overlay_array[full_size_mask] = (
        0.45 * original_array[full_size_mask]
        + 0.55 * red[full_size_mask]
    ).astype(np.uint8)

    overlay_image = Image.fromarray(overlay_array)

    return coverage, mask_image, overlay_image

# =========================================================
# MANAGEMENT INDICATOR
# =========================================================

def get_management_indicator(coverage):
    if coverage < 10:
        return "Low Weed Presence"
    elif coverage < 25:
        return "Moderate Weed Presence"
    else:
        return "High Weed Presence"

# =========================================================
# SINGLE-IMAGE SUGGESTIONS
# =========================================================

def show_single_suggestions(coverage):
    st.divider()
    st.subheader("🌿 Smart Weed Management Suggestions")

    if coverage < 10:
        st.success("Low predicted foreground coverage")

        st.write(
            "- Continue checking the field regularly for new weed growth."
        )
        st.write(
            "- Maintain preventive practices, such as keeping field "
            "edges and irrigation channels under observation."
        )
        st.write(
            "- Recheck the field after suitable intervals, especially "
            "following rainfall or irrigation."
        )

    elif coverage < 25:
        st.warning("Moderate predicted foreground coverage")

        st.write(
            "- Inspect the red-highlighted regions to locate areas "
            "that may need attention."
        )
        st.write(
            "- Where practical, remove or manage weeds early before "
            "they spread or produce seeds."
        )
        st.write(
            "- Monitor the same field areas again to check whether "
            "the predicted coverage is increasing."
        )

    else:
        st.error("High predicted foreground coverage")

        st.write(
            "- Prioritize a field inspection, focusing on the "
            "regions highlighted in red."
        )
        st.write(
            "- Identify the weed species and assess the crop and "
            "growth stage before choosing a control method."
        )
        st.write(
            "- Consider suitable integrated weed management options, "
            "such as mechanical or cultural control, with advice "
            "from a qualified agricultural professional."
        )

    st.caption(
        "These suggestions are general guidance based on predicted "
        "pixel coverage, not confirmed weed measurements. The model "
        "does not identify weed species or prescribe treatments."
    )

# =========================================================
# BEFORE-AFTER SUGGESTIONS
# =========================================================

def show_comparison_suggestions(coverage_change):
    st.divider()
    st.subheader("🌿 Smart Management Suggestions")

    if coverage_change > 0:
        st.success(
            "Good sign: predicted foreground coverage has decreased."
        )

        st.write(
            "- Continue monitoring the field to see whether the "
            "decrease is maintained."
        )
        st.write(
            "- Continue appropriate field-management practices "
            "that are already working well."
        )
        st.write(
            "- Take follow-up images from a similar viewpoint and "
            "under similar lighting conditions for a fair comparison."
        )

    elif coverage_change < 0:
        st.warning(
            "Attention needed: predicted foreground coverage has increased."
        )

        st.write(
            "- Inspect the red-highlighted regions to check for "
            "possible new or spreading weed growth."
        )
        st.write(
            "- Verify that both images show comparable areas and "
            "were taken under similar conditions."
        )
        st.write(
            "- Identify the weeds and consider timely mechanical "
            "or cultural control where appropriate."
        )
        st.write(
            "- If the increase is confirmed, seek local agricultural "
            "advice about suitable next steps."
        )

    else:
        st.info(
            "Predicted foreground coverage has remained the same."
        )

        st.write(
            "- Continue regular monitoring rather than assuming "
            "that weed growth has stopped."
        )
        st.write(
            "- Compare future images of the same field area to "
            "check for changes over time."
        )
        st.write(
            "- Inspect the field directly if weeds appear to be "
            "spreading despite similar model predictions."
        )

    st.caption(
        "A change in predicted coverage does not prove that weeds "
        "increased or decreased in reality, or that a treatment "
        "caused the change. Confirm results through field inspection."
    )

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("🌱 Project")

    st.write(
        "AI-powered agricultural image analysis "
        "using a trained Attention U-Net."
    )

    st.divider()
    st.subheader("🤖 Model Information")

    st.write("**Architecture:** Attention U-Net")
    st.write("**Task:** Binary image segmentation")
    st.write("**Input Size:** 256 × 256")
    st.write("**Loss:** Focal + Dice")
    st.write("**Optimizer:** Adam")
    st.write(f"**Threshold:** {best_threshold:.3f}")

    st.divider()
    st.subheader("📌 Features")

    st.write("✓ Binary segmentation mask")
    st.write("✓ Predicted foreground coverage")
    st.write("✓ Red-highlighted overlay")
    st.write("✓ Before vs After comparison")
    st.write("✓ Management suggestions")

# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🌱 AI-Based Precision Weed Management'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep learning-based weed segmentation and field analysis'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.success(
    "Attention U-Net loaded successfully. "
    f"Prediction threshold: {best_threshold:.3f}"
)

# =========================================================
# SELECT ANALYSIS MODE
# =========================================================

st.markdown(
    '<div class="section-heading">'
    'Choose Analysis Mode'
    '</div>',
    unsafe_allow_html=True
)

mode_col1, mode_col2 = st.columns(2)

with mode_col1:
    st.info(
        "**🔍 Single Image Analysis**\n\n"
        "Analyze one field image and get suggestions."
    )

with mode_col2:
    st.info(
        "**📊 Before vs After Comparison**\n\n"
        "Compare two field images and get suggestions."
    )

analysis_mode = st.radio(
    "Select analysis type",
    [
        "Single Image Analysis",
        "Before vs After Comparison"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

if st.session_state.previous_mode != analysis_mode:
    st.session_state.single_result = None
    st.session_state.single_result_key = None
    st.session_state.comparison_result = None
    st.session_state.comparison_result_key = None
    st.session_state.previous_mode = analysis_mode

# =========================================================
# SINGLE IMAGE ANALYSIS
# =========================================================

if analysis_mode == "Single Image Analysis":

    st.divider()
    st.markdown(
        '<div class="section-heading">'
        '📷 Analyze Field Image'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload an image and click Analyze Image to run the model."
    )

    uploaded_file = st.file_uploader(
        "Choose a field image",
        type=["jpg", "jpeg", "png"],
        key="single_image"
    )

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            image_key = hashlib.md5(image_bytes).hexdigest()

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.subheader("🖼️ Original Field Image")
            st.image(
                image,
                caption="Uploaded Field Image",
                width=500
            )

            if st.session_state.single_result_key != image_key:
                st.session_state.single_result = None
                st.session_state.single_result_key = image_key

            if st.button(
                "🔍 Analyze Image",
                type="primary",
                key="analyze_single"
            ):
                with st.spinner("Analyzing field image..."):
                    coverage, mask_image, overlay_image = (
                        analyze_image(image)
                    )

                st.session_state.single_result = {
                    "coverage": coverage,
                    "mask": mask_image,
                    "overlay": overlay_image
                }

            result = st.session_state.single_result

            if (
                result is not None
                and st.session_state.single_result_key == image_key
            ):
                coverage = result["coverage"]
                indicator = get_management_indicator(coverage)

                st.divider()
                st.subheader("🧠 Segmentation Results")

                image_col1, image_col2 = st.columns(2)

                with image_col1:
                    st.image(
                        result["mask"],
                        caption="Predicted Binary Mask",
                        width=350
                    )

                with image_col2:
                    st.image(
                        result["overlay"],
                        caption="Predicted Mask Overlay",
                        width=350
                    )

                st.divider()
                st.subheader("📊 Analysis Results")

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "🌱 Predicted Foreground Coverage",
                        f"{coverage:.1f}%"
                    )

                with metric_col2:
                    st.metric(
                        "📌 Management Indicator",
                        indicator
                    )

                # NEW: Suggestions for a single image
                show_single_suggestions(coverage)

        except Exception as error:
            st.error("Unable to analyze this image.")
            st.exception(error)

    else:
        st.session_state.single_result = None
        st.session_state.single_result_key = None

# =========================================================
# BEFORE VS AFTER COMPARISON
# =========================================================

else:

    st.divider()
    st.markdown(
        '<div class="section-heading">'
        '📊 Before vs After Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload two comparable images and click Compare Field Images."
    )

    before_col, after_col = st.columns(2)

    before_image = None
    after_image = None

    with before_col:
        st.subheader("🌱 Previous / Before")

        before_file = st.file_uploader(
            "Upload previous field image",
            type=["jpg", "jpeg", "png"],
            key="before_image"
        )

        if before_file is not None:
            try:
                before_image = Image.open(
                    before_file
                ).convert("RGB")

                st.image(
                    before_image,
                    caption="Previous Field Image",
                    width=350
                )

            except Exception:
                st.error("Unable to open the before image.")

    with after_col:
        st.subheader("🌿 Current / After")

        after_file = st.file_uploader(
            "Upload current field image",
            type=["jpg", "jpeg", "png"],
            key="after_image"
        )

        if after_file is not None:
            try:
                after_image = Image.open(
                    after_file
                ).convert("RGB")

                st.image(
                    after_image,
                    caption="Current Field Image",
                    width=350
                )

            except Exception:
                st.error("Unable to open the after image.")

    comparison_key = None

    if before_file is not None and after_file is not None:
        before_hash = hashlib.md5(
            before_file.getvalue()
        ).hexdigest()

        after_hash = hashlib.md5(
            after_file.getvalue()
        ).hexdigest()

        comparison_key = before_hash + "_" + after_hash

    if (
        st.session_state.comparison_result_key
        != comparison_key
    ):
        st.session_state.comparison_result = None
        st.session_state.comparison_result_key = comparison_key

    if before_image is not None and after_image is not None:

        if st.button(
            "📊 Compare Field Images",
            type="primary",
            key="compare_images"
        ):
            try:
                with st.spinner("Analyzing both field images..."):
                    (
                        before_coverage,
                        before_mask,
                        before_overlay
                    ) = analyze_image(before_image)

                    (
                        after_coverage,
                        after_mask,
                        after_overlay
                    ) = analyze_image(after_image)

                # Positive: coverage decreased.
                # Negative: coverage increased.
                coverage_change = (
                    before_coverage - after_coverage
                )

                st.session_state.comparison_result = {
                    "before_coverage": before_coverage,
                    "after_coverage": after_coverage,
                    "coverage_change": coverage_change,
                    "before_mask": before_mask,
                    "after_mask": after_mask,
                    "before_overlay": before_overlay,
                    "after_overlay": after_overlay
                }

            except Exception as error:
                st.error("Unable to compare the images.")
                st.exception(error)

    elif before_file is not None or after_file is not None:
        st.info(
            "Upload both the before and after images "
            "to enable comparison."
        )

    comparison = st.session_state.comparison_result

    if (
        comparison is not None
        and comparison_key is not None
        and st.session_state.comparison_result_key == comparison_key
    ):
        before_coverage = comparison["before_coverage"]
        after_coverage = comparison["after_coverage"]
        coverage_change = comparison["coverage_change"]

        st.divider()
        st.subheader("📈 Weed Coverage Comparison")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Before Coverage",
                f"{before_coverage:.1f}%"
            )

        with result_col2:
            st.metric(
                "After Coverage",
                f"{after_coverage:.1f}%"
            )

        with result_col3:
            st.metric(
                "Absolute Change",
                f"{abs(coverage_change):.1f} percentage points"
            )

        if coverage_change > 0:
            st.success(
                "🌱 Predicted foreground coverage decreased "
                f"by {coverage_change:.1f} percentage points."
            )

        elif coverage_change < 0:
            st.warning(
                "⚠️ Predicted foreground coverage increased "
                f"by {abs(coverage_change):.1f} percentage points."
            )

        else:
            st.info(
                "No change in predicted foreground coverage."
            )

        st.divider()
        st.subheader("🧠 Segmentation Masks")

        mask_col1, mask_col2 = st.columns(2)

        with mask_col1:
            st.write("**Before — Predicted Mask**")
            st.image(comparison["before_mask"], width=350)

        with mask_col2:
            st.write("**After — Predicted Mask**")
            st.image(comparison["after_mask"], width=350)

        st.divider()
        st.subheader("🔴 Predicted Foreground Overlays")

        overlay_col1, overlay_col2 = st.columns(2)

        with overlay_col1:
            st.write("**Before — Overlay**")
            st.image(comparison["before_overlay"], width=350)

        with overlay_col2:
            st.write("**After — Overlay**")
            st.image(comparison["after_overlay"], width=350)

        # NEW: Suggestions based on the coverage change
        show_comparison_suggestions(coverage_change)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Based Precision Weed Management • "
    "Attention U-Net • TensorFlow/Keras"
)