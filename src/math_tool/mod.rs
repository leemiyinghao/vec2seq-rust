use glam::Vec4;

pub fn consine_similarity(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Result<f32, &'static str> {
    if vec1.len() != vec2.len() {
        return Err("length of vectors does not equal");
    } else {
        let x_norm = dot(vec1, vec1).unwrap_or(0f32);
        let y_norm = dot(vec2, vec2).unwrap_or(0f32);
        let x_y_norm = x_norm * y_norm;
        if x_y_norm == 0f32 {
            //one of vec have 0 on norm, fallback to 0.0
            Ok(0.0f32)
        } else {
            Ok(1.0f32 * dot(vec1, vec2).unwrap_or(0f32) / x_y_norm.sqrt())
        }
    }
}
pub fn dot(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Result<f32, &'static str> {
    if vec1.len() != vec2.len() {
        return Err("length of vectors does not equal");
    } else {
        let mut result: f32 = 0f32;
        for i in 0..(vec1.len() / 4) {
            result += Vec4::new(
                vec1[&i * 4 + 0],
                vec1[&i * 4 + 1],
                vec1[&i * 4 + 2],
                vec1[&i * 4 + 3],
            )
            .dot(Vec4::new(
                vec2[&i * 4 + 0],
                vec2[&i * 4 + 1],
                vec2[&i * 4 + 2],
                vec2[&i * 4 + 3],
            ))
        }
        let rest = (vec1.len() % 4) + 0;
        Ok({
            let start: usize = vec1.len() - rest;
            let (x1, y1, x2, y2, x3, y3) = match rest {
                1usize => (vec1[start], vec2[start], 0f32, 0f32, 0f32, 0f32),
                2usize => (
                    vec1[start],
                    vec2[start],
                    vec1[start + 1],
                    vec2[start + 1],
                    0f32,
                    0f32,
                ),
                3usize => (
                    vec1[start],
                    vec2[start],
                    vec1[start + 1],
                    vec2[start + 1],
                    vec1[start + 2],
                    vec2[start + 2],
                ),
                0usize | _ => (0f32, 0f32, 0f32, 0f32, 0f32, 0f32),
            };
            &result + Vec4::new(x1, x2, x3, 0f32).dot(Vec4::new(y1, y2, y3, 0f32))
        })
    }
}
