# Openshorts Layout Engineering Notes

This document exists to preserve the exact mathematical relationships and architectural choices that connect the **React UI Live Preview** (CSS) with the **Backend FFmpeg generation** (`app.py`).

The core problem solved here is gracefully translating arbitrary video geometries (like 16:9 vertical clips or 4:3) into arbitrary target containers (like 9:16 vertical shorts) while allowing the user to zoom and pan **both the aspect-ratio overhang and the zoom overhang**.

## 1. The FFmpeg Logic (`app.py`)
When rendering the final layout via standard `cover` or `contain` modes, FFmpeg uses three distinct operations:

1. **Calculate Capture Box**: We determine precisely how much of the original frame we want to capture. If `base_scale` fits the container, and the user zooms by `Z` (e.g. 1.5x), the capture box is `Z` times smaller than the source image.
2. **Crop (Pan step)**: If the capture box is smaller than the original video in either width or height, we apply a `crop` filter. The user's `offset_x` and `offset_y` inputs (-100 to 100) are mapped linearly across the maximum crop space, allowing panning through both intrinsic aspect-ratio overhang and the new space created by zooming.
3. **Scale**: The cropped box is then scaled via interpolation to either touch the container's edges (cover) or fit exactly inside (contain).
4. **Pad (Contain Pan step)**: If the scaled image is *smaller* than the container, we use a `pad` filter with black background. Any offsets are now used to push the image around the black box.

*(Important: Pan offsets never double-dip. They apply to Crop OR Pad on a given axis, never both simultaneously since a crop implies the image touches the container edge).*

## 2. The React CSS Magic (`ClipStudioModal.jsx`)
To avoid laggy server-side previews, the frontend perfectly mimics the FFmpeg operations instantaneously using native CSS, without actually modifying the source clip.

### The Problem with CSS
Standard CSS `object-fit: cover` maps intrinsic aspect-ratio overhang beautifully using `object-position`. However, it breaks when the user zooms in via CSS `transform: scale()`, because `object-position` only maps coordinates within the unscaled bounding box. Thus, when zoomed, you historically couldn't pan to the newly hidden edges (e.g. vertical panning in a 9:16 layout).

### The Solution: CSS Math Alignment
To achieve 1:1 parity with FFmpeg, the frontend combines `object-position` and `transform: translate`:

1. **Intrinsic Overhang (`object-position`)**:
   We map the incoming `-100 to 100` slider ranges to `0% to 100%`:
   ```javascript
   const posX = 50 + (effectX / 2);
   const posY = 50 + (effectY / 2);
   // Used in manualLayoutObjectPosition
   ```
   This handles panning edge-to-edge if the raw video spills outside the container natively (e.g. 16:9 in a 9:16 container spills horizontally).

2. **Zoom Overhang (`transform: translate`)**:
   When `layoutZoom` (Z) is > 1.0, the element becomes larger than the container, hiding pixels pushed over the edge. The maximum translation percentage to reveal an edge without overshooting into black space is exactly `(Z - 1) * 50%` of the element.
   We deduce the exact translation required:
   ```javascript
   const transX = -(effectX / 100) * ((zNum - 1) / 2) * 100;
   const transY = -(effectY / 100) * ((zNum - 1) / 2) * 100;
   // Final Transform: scale(Z) translate(transX%, transY%)
   ```

Because CSS separates `object-position` (handling the intrinsic ratio overflow safely) and `translate` (handling the new scale overflow), this mathematical combo behaves exactly like FFmpeg's unified calculation, enabling pixel-perfect panning across both X and Y axes in all modes.

### 3. Contain Workflow
The `Contain` mode behaves uniquely.
Instead of complicated CSS letterboxing math, the frontend simply mimics `Contain` by using `object-fit: cover` and zooming *out* (Z < 1.0).
- If the user selects *Contain*, the minimum zoom limit is dropped to `0.3`, allowing them to visually shrink the video back down into view.
- The `zNum < 1` calculation automatically flips the signs on `transX` and `transY`, gracefully allowing the image to be panned seamlessly inside the letterbox padding!

**NEVER alter these equations isolated from each other.** CSS `transform`, `object-fit`, and `app.py` crop calculations are inextricably linked.
