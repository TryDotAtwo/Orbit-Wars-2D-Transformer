use crate::config::AMOUNT_CLASS_COUNT;

pub const NO_PLANET_ID: i32 = -1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Planet {
    pub id: i32,
    pub owner: i32,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub ships: f32,
    pub production: f32,
    pub velocity_x: f32,
    pub velocity_y: f32,
}

impl Planet {
    pub fn distance_squared_to(&self, other: &Planet) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fleet {
    pub id: i32,
    pub owner: i32,
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub from_planet_id: i32,
    pub ships: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveCommand {
    pub from_planet_id: i32,
    pub direction_angle: f32,
    pub ship_count: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AmountClass {
    AllIn = 0,
    Percent25 = 1,
    Percent50 = 2,
    Percent67 = 3,
    Percent70 = 4,
    Percent75 = 5,
    Percent90 = 6,
    Percent95 = 7,
    Ships10 = 8,
    Ships20 = 9,
    Ships50 = 10,
    Ships100 = 11,
    Ships200 = 12,
    Ships500 = 13,
    Ships1000 = 14,
    Ships2000 = 15,
}

impl AmountClass {
    pub fn from_index(index: usize) -> Option<Self> {
        Some(match index {
            0 => Self::AllIn,
            1 => Self::Percent25,
            2 => Self::Percent50,
            3 => Self::Percent67,
            4 => Self::Percent70,
            5 => Self::Percent75,
            6 => Self::Percent90,
            7 => Self::Percent95,
            8 => Self::Ships10,
            9 => Self::Ships20,
            10 => Self::Ships50,
            11 => Self::Ships100,
            12 => Self::Ships200,
            13 => Self::Ships500,
            14 => Self::Ships1000,
            15 => Self::Ships2000,
            _ => return None,
        })
    }

    pub fn ship_count(self, source_ships: f32) -> i32 {
        let requested = match self {
            Self::AllIn => source_ships,
            Self::Percent25 => source_ships * 0.25,
            Self::Percent50 => source_ships * 0.50,
            Self::Percent67 => source_ships * 0.67,
            Self::Percent70 => source_ships * 0.70,
            Self::Percent75 => source_ships * 0.75,
            Self::Percent90 => source_ships * 0.90,
            Self::Percent95 => source_ships * 0.95,
            Self::Ships10 => 10.0,
            Self::Ships20 => 20.0,
            Self::Ships50 => 50.0,
            Self::Ships100 => 100.0,
            Self::Ships200 => 200.0,
            Self::Ships500 => 500.0,
            Self::Ships1000 => 1000.0,
            Self::Ships2000 => 2000.0,
        };
        requested.min(source_ships).floor().max(0.0) as i32
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionSlotOutput {
    pub fire_logit: f32,
    pub source_logits: [f32; 64],
    pub target_logits: [f32; 64],
    pub amount_logits: [f32; AMOUNT_CLASS_COUNT],
}
