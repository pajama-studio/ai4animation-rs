pub(crate) type Vec3 = [f32; 3];

pub(crate) fn add(left: Vec3, right: Vec3) -> Vec3 {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

pub(crate) fn sub(left: Vec3, right: Vec3) -> Vec3 {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

pub(crate) fn scale(vector: Vec3, amount: f32) -> Vec3 {
    [vector[0] * amount, vector[1] * amount, vector[2] * amount]
}

pub(crate) fn dot(left: Vec3, right: Vec3) -> f32 {
    left[0].mul_add(right[0], left[1].mul_add(right[1], left[2] * right[2]))
}

pub(crate) fn cross(left: Vec3, right: Vec3) -> Vec3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

pub(crate) fn length(vector: Vec3) -> f32 {
    dot(vector, vector).sqrt()
}

pub(crate) fn normalize_or(vector: Vec3, fallback: Vec3) -> Vec3 {
    let magnitude = length(vector);
    if magnitude > 1.0e-7 && magnitude.is_finite() {
        scale(vector, magnitude.recip())
    } else {
        fallback
    }
}

pub(crate) fn distance(left: Vec3, right: Vec3) -> f32 {
    length(sub(left, right))
}

pub(crate) fn lerp(left: Vec3, right: Vec3, amount: f32) -> Vec3 {
    add(left, scale(sub(right, left), amount))
}
