use crate::config::AgentConfig;

pub fn ship_log_percent(ships: f32, config: &AgentConfig) -> f32 {
    let normalized_ships = ships.max(0.0);
    if normalized_ships <= config.ship_log_pivot {
        return 0.9 * (1.0 + normalized_ships).ln() / (1.0 + config.ship_log_pivot).ln();
    }

    let upper = (normalized_ships / config.ship_log_pivot).ln()
        / (config.ship_log_cap / config.ship_log_pivot).ln();
    0.9 + 0.1 * upper.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ship_scaler_maps_zero_to_zero() {
        let config = AgentConfig::default();
        assert_eq!(ship_log_percent(0.0, &config), 0.0);
    }

    #[test]
    fn ship_scaler_maps_pivot_to_ninety_percent() {
        let config = AgentConfig::default();
        let scaled = ship_log_percent(config.ship_log_pivot, &config);
        assert!((scaled - 0.9).abs() < 0.000_001);
    }

    #[test]
    fn ship_scaler_clamps_at_cap() {
        let config = AgentConfig::default();
        let scaled = ship_log_percent(config.ship_log_cap * 2.0, &config);
        assert!((scaled - 1.0).abs() < 0.000_001);
    }
}
