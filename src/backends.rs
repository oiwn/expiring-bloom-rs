// use bitvec::prelude::*;
use std::time::SystemTime;

// Trait for the storage backend
pub trait BloomFilterStorage {
    fn new(capacity: usize, max_levels: usize) -> Self;
    fn set_bit(&mut self, level: usize, index: usize);
    fn get_bit(&self, level: usize, index: usize) -> bool;
    fn clear_level(&mut self, level: usize);
    fn set_timestamp(&mut self, level: usize, timestamp: SystemTime);
    fn get_timestamp(&self, level: usize) -> Option<SystemTime>;
    fn num_levels(&self) -> usize;
}

// In-memory storage implementation
pub struct InMemoryStorage {
    pub levels: Vec<Vec<bool>>,
    timestamps: Vec<SystemTime>,
    capacity: usize,
}

impl BloomFilterStorage for InMemoryStorage {
    fn new(capacity: usize, max_levels: usize) -> Self {
        Self {
            levels: vec![vec![false; capacity]; max_levels],
            timestamps: vec![SystemTime::now(); max_levels],
            capacity,
        }
    }

    fn set_bit(&mut self, level: usize, index: usize) {
        self.levels[level][index] = true;
    }

    fn get_bit(&self, level: usize, index: usize) -> bool {
        self.levels[level][index]
    }

    fn clear_level(&mut self, level: usize) {
        self.levels[level] = vec![false; self.capacity];
    }

    fn set_timestamp(&mut self, level: usize, timestamp: SystemTime) {
        self.timestamps[level] = timestamp;
    }

    fn get_timestamp(&self, level: usize) -> Option<SystemTime> {
        Some(self.timestamps[level])
    }

    fn num_levels(&self) -> usize {
        self.levels.len()
    }
}
