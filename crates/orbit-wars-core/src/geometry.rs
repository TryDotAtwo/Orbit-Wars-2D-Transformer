use crate::config::AgentConfig;
use crate::types::Planet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryError {
    NonFiniteCoordinate,
    NonPositiveSpeed,
}

#[derive(Clone, Copy, Debug)]
pub struct OrbitPrediction {
    pub center_x: f32,
    pub center_y: f32,
    pub angular_velocity: f32,
}

pub fn fleet_speed(ship_count: f32, config: &AgentConfig) -> f32 {
    let ships = ship_count.max(1.0);
    let speed_ratio = (ships.ln() / config.fleet_speed_reference_ships.ln()).clamp(0.0, 1.0);
    1.0 + (config.fleet_speed_max - 1.0) * speed_ratio.powf(config.fleet_speed_curve_power)
}

pub fn intercept_angle(
    source: &Planet,
    target: &Planet,
    ship_count: f32,
    config: &AgentConfig,
) -> Result<f32, GeometryError> {
    intercept_angle_with_prediction(source, target, ship_count, config, None)
}

pub fn intercept_angle_with_orbit_prediction(
    source: &Planet,
    target: &Planet,
    ship_count: f32,
    config: &AgentConfig,
    orbit: OrbitPrediction,
) -> Result<f32, GeometryError> {
    intercept_angle_with_prediction(source, target, ship_count, config, Some(orbit))
}

fn intercept_angle_with_prediction(
    source: &Planet,
    target: &Planet,
    ship_count: f32,
    config: &AgentConfig,
    orbit: Option<OrbitPrediction>,
) -> Result<f32, GeometryError> {
    if !(source.x.is_finite()
        && source.y.is_finite()
        && target.x.is_finite()
        && target.y.is_finite()
        && target.velocity_x.is_finite()
        && target.velocity_y.is_finite()
        && orbit
            .map(|motion| {
                motion.center_x.is_finite()
                    && motion.center_y.is_finite()
                    && motion.angular_velocity.is_finite()
            })
            .unwrap_or(true))
    {
        return Err(GeometryError::NonFiniteCoordinate);
    }

    let speed = fleet_speed(ship_count, config);
    if speed <= 0.0 {
        return Err(GeometryError::NonPositiveSpeed);
    }

    let mut predicted_x = target.x;
    let mut predicted_y = target.y;
    let contact_gap = source.radius + config.fleet_spawn_offset + target.radius;
    for _ in 0..config.intercept_iterations {
        let dx = predicted_x - source.x;
        let dy = predicted_y - source.y;
        let distance = (dx * dx + dy * dy).sqrt();
        let travel_time = ((distance - contact_gap) / speed).max(0.0);
        let predicted = predict_target_position(target, travel_time, orbit);
        predicted_x = predicted.0;
        predicted_y = predicted.1;
    }

    Ok((predicted_y - source.y).atan2(predicted_x - source.x))
}

fn predict_target_position(
    target: &Planet,
    travel_time: f32,
    orbit: Option<OrbitPrediction>,
) -> (f32, f32) {
    if let Some(motion) = orbit {
        let dx = target.x - motion.center_x;
        let dy = target.y - motion.center_y;
        let radius = (dx * dx + dy * dy).sqrt();
        if radius > f32::EPSILON {
            let angle = dy.atan2(dx) + motion.angular_velocity * travel_time;
            return (
                motion.center_x + radius * angle.cos(),
                motion.center_y + radius * angle.sin(),
            );
        }
    }
    (
        target.x + target.velocity_x * travel_time,
        target.y + target.velocity_y * travel_time,
    )
}

pub fn segment_intersects_sun(
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
    config: &AgentConfig,
) -> bool {
    segment_distance_squared_to_point(
        start_x,
        start_y,
        end_x,
        end_y,
        config.board_center,
        config.board_center,
    ) < config.sun_radius * config.sun_radius
}

pub fn segment_intersects_circle(
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
    center_x: f32,
    center_y: f32,
    radius: f32,
) -> bool {
    let dx = end_x - start_x;
    let dy = end_y - start_y;
    let length_squared = dx * dx + dy * dy;
    if length_squared == 0.0 {
        let point_dx = start_x - center_x;
        let point_dy = start_y - center_y;
        return point_dx * point_dx + point_dy * point_dy <= radius * radius;
    }

    let projection =
        (((center_x - start_x) * dx + (center_y - start_y) * dy) / length_squared).clamp(0.0, 1.0);
    let closest_x = start_x + projection * dx;
    let closest_y = start_y + projection * dy;
    let closest_dx = closest_x - center_x;
    let closest_dy = closest_y - center_y;
    closest_dx * closest_dx + closest_dy * closest_dy <= radius * radius
}

fn segment_distance_squared_to_point(
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
    point_x: f32,
    point_y: f32,
) -> f32 {
    let dx = end_x - start_x;
    let dy = end_y - start_y;
    let length_squared = dx * dx + dy * dy;
    if length_squared == 0.0 {
        let point_dx = start_x - point_x;
        let point_dy = start_y - point_y;
        return point_dx * point_dx + point_dy * point_dy;
    }

    let projection =
        (((point_x - start_x) * dx + (point_y - start_y) * dy) / length_squared).clamp(0.0, 1.0);
    let closest_x = start_x + projection * dx;
    let closest_y = start_y + projection * dy;
    let closest_dx = closest_x - point_x;
    let closest_dy = closest_y - point_y;
    closest_dx * closest_dx + closest_dy * closest_dy
}

pub fn segment_circle_hit_fraction(
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
    center_x: f32,
    center_y: f32,
    radius: f32,
) -> Option<f32> {
    let dx = end_x - start_x;
    let dy = end_y - start_y;
    let fx = start_x - center_x;
    let fy = start_y - center_y;
    let a = dx * dx + dy * dy;
    if a == 0.0 {
        let inside = fx * fx + fy * fy <= radius * radius;
        return if inside { Some(0.0) } else { None };
    }
    let b = 2.0 * (fx * dx + fy * dy);
    let c = fx * fx + fy * fy - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_discriminant = discriminant.sqrt();
    let entry = (-b - sqrt_discriminant) / (2.0 * a);
    let exit = (-b + sqrt_discriminant) / (2.0 * a);
    if (0.0..=1.0).contains(&entry) {
        Some(entry)
    } else if (0.0..=1.0).contains(&exit) {
        Some(exit)
    } else {
        None
    }
}

pub fn swept_pair_hit(
    fleet_start_x: f32,
    fleet_start_y: f32,
    fleet_end_x: f32,
    fleet_end_y: f32,
    planet_start_x: f32,
    planet_start_y: f32,
    planet_end_x: f32,
    planet_end_y: f32,
    radius: f32,
) -> bool {
    let delta_start_x = fleet_start_x - planet_start_x;
    let delta_start_y = fleet_start_y - planet_start_y;
    let delta_velocity_x = (fleet_end_x - fleet_start_x) - (planet_end_x - planet_start_x);
    let delta_velocity_y = (fleet_end_y - fleet_start_y) - (planet_end_y - planet_start_y);
    let a = delta_velocity_x * delta_velocity_x + delta_velocity_y * delta_velocity_y;
    let b = 2.0 * (delta_start_x * delta_velocity_x + delta_start_y * delta_velocity_y);
    let c = delta_start_x * delta_start_x + delta_start_y * delta_start_y - radius * radius;
    if a < f32::EPSILON {
        return c <= 0.0;
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return false;
    }
    let root = discriminant.sqrt();
    let entry = (-b - root) / (2.0 * a);
    let exit = (-b + root) / (2.0 * a);
    exit >= 0.0 && entry <= 1.0
}

pub fn point_out_of_bounds(x: f32, y: f32, config: &AgentConfig) -> bool {
    x < 0.0 || y < 0.0 || x > config.board_size || y > config.board_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_ship_speed_is_one() {
        let config = AgentConfig::default();
        assert!((fleet_speed(1.0, &config) - 1.0).abs() < 0.000_001);
    }

    #[test]
    fn sun_collision_detects_segment_crossing_center() {
        let config = AgentConfig::default();
        assert!(segment_intersects_sun(0.0, 50.0, 100.0, 50.0, &config));
    }

    #[test]
    fn sun_collision_rejects_segment_far_from_center() {
        let config = AgentConfig::default();
        assert!(!segment_intersects_sun(0.0, 0.0, 100.0, 0.0, &config));
    }

    #[test]
    fn sun_tangent_segment_is_not_collision() {
        let config = AgentConfig::default();
        assert!(!segment_intersects_sun(40.0, 40.0, 60.0, 40.0, &config));
    }

    #[test]
    fn segment_circle_hit_fraction_returns_first_contact() {
        let hit = segment_circle_hit_fraction(0.0, 0.0, 10.0, 0.0, 5.0, 0.0, 1.0).unwrap();
        assert!((hit - 0.4).abs() < 0.000_001);
    }

    #[test]
    fn swept_pair_hit_detects_crossing_moving_objects() {
        assert!(swept_pair_hit(
            49.0, 50.0, 51.0, 50.0, 50.0, 52.0, 50.0, 48.0, 1.0,
        ));
    }
}
