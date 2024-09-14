#![allow(non_upper_case_globals)]
/// This is implementation of Silding Bloom Filter.
///
/// Features:
///    * Sub-Filters: The main Bloom filter is divided into N sub-filters:  BF_1, BF_2, …, BF_N .
///    * Time Windows: Each sub-filter corresponds to a fixed time window  T  (e.g., 1 minute).
///    * Rotation Mechanism: Sub-filters are rotated in a circular manner to represent sliding
///      time intervals.
///
/// Insertion:
///     * When an element is added at time  t , it is inserted into the current sub-filter  BF_{current}.
///     * Hash the element using the standard Bloom filter hash functions and set the bits in  BF_{current} .
/// Query:
///     * To check if an element is in the filter, perform the query against all active sub-filters.
///     * If all the required bits are set in any sub-filter, the element is considered present.
/// Expiration:
///     * At each time interval  T , the oldest sub-filter  BF_{oldest}  is cleared.
///     * The cleared sub-filter becomes the new  BF_{current}  for incoming elements.
///     * This effectively removes elements that were only in  BF_{oldest} , thus expiring them.
///
/// Obvious problems:
///     * False Positives: As elements may exist in multiple sub-filters,
///       the probability of false positives can increase.
///     * Synchronization: In concurrent environments, care must be taken to synchronize
///       access during sub-filter rotation.
use bitvec::prelude::*;
use fnv::FnvHasher;
use murmur3::murmur3_32;
use std::hash::Hasher;
use std::io::Cursor;
use std::time::{Duration, SystemTime};

// Trait for the bit vector backend
pub trait BitVector {
    fn new(size: usize) -> Self;
    fn set(&mut self, index: usize);
    fn get(&self, index: usize) -> bool;
    fn clear(&mut self);
}

/// A type alias for the hash function used in the Bloom filter.
///
/// This function takes an input item and computes multiple hash indices
/// for the Bloom filter's bit vector.
///
/// **Parameters:**
///
/// - `item: &[u8]`
///   - A byte slice representing the item to be hashed.
/// - `num_hashes: usize`
///   - The number of hash values to compute for the item.
/// - `capacity: usize`
///   - The size of the Bloom filter's bit vector. This ensures that
///     the generated hash indices are within valid bounds.
///
/// **Returns:**
///
/// - `Vec<u32>`
///   - A vector of hash indices corresponding to positions in the bit vector.
///
/// **Usage:**
///
/// The hash function computes `num_hashes` hash indices for the given `item`,
/// ensuring each index is within the range `[0, capacity)`. These indices are
/// used to set or check bits in the Bloom filter's bit vector.
type HashFunction = fn(&[u8], usize, usize) -> Vec<u32>;

fn hash_murmur32(key: &[u8]) -> u32 {
    let mut cursor = Cursor::new(key);
    murmur3_32(&mut cursor, 0).expect("Failed to compute Murmur3 hash")
}

fn hash_fnv32(key: &[u8]) -> u32 {
    let mut hasher = FnvHasher::default();
    hasher.write(key);
    hasher.finish() as u32
}

pub fn default_hash_function(
    item: &[u8],
    num_hashes: usize,
    capacity: usize,
) -> Vec<u32> {
    let h1 = hash_murmur32(item);
    let h2 = hash_fnv32(item);
    (0..num_hashes)
        .map(|i| h1.wrapping_add((i as u32).wrapping_mul(h2)) % capacity as u32)
        .collect()
}

// Structure for a single Bloom filter level
pub struct SlidingBloomFilterLevel<B: BitVector> {
    bit_vector: B,
    timestamp: SystemTime,
}

pub struct SlidingBloomFilter<B: BitVector> {
    levels: Vec<SlidingBloomFilterLevel<B>>,
    hash_function: HashFunction,
    capacity: usize,
    num_hashes: usize,
    false_positive_rate: f64,
    bit_vector_size: usize,
    level_time: Duration,
    max_levels: usize,
}

fn optimal_bit_vector_size(n: usize, fpr: f64) -> usize {
    let ln2 = std::f64::consts::LN_2;
    ((-(n as f64) * fpr.ln()) / (ln2 * ln2)).ceil() as usize
}

fn optimal_num_hashes(n: usize, m: usize) -> usize {
    ((m as f64 / n as f64) * std::f64::consts::LN_2).round() as usize
}

impl<B: BitVector> SlidingBloomFilter<B> {
    pub fn new(
        capacity: usize,
        false_positive_rate: f64,
        level_time: Duration,
        max_levels: usize,
        hash_function: HashFunction,
    ) -> Self {
        let bit_vector_size =
            optimal_bit_vector_size(capacity, false_positive_rate);
        let num_hashes = optimal_num_hashes(capacity, bit_vector_size);
        Self {
            levels: Vec::new(),
            hash_function,
            capacity,
            bit_vector_size,
            num_hashes,
            false_positive_rate,
            level_time,
            max_levels,
        }
    }

    pub fn cleanup_expired_levels(&mut self) {
        let now = SystemTime::now();
        self.levels.retain(|level| {
            match now.duration_since(level.timestamp) {
                Ok(elapsed) => elapsed < self.level_time * self.max_levels as u32,
                Err(_) => true, // Keep the level if system time went backwards
            }
        });
    }

    fn should_create_new_level(&self) -> bool {
        if let Some(last_level) = self.levels.last() {
            let now = SystemTime::now();
            match now.duration_since(last_level.timestamp) {
                Ok(elapsed) => elapsed >= self.level_time,
                Err(_) => false, // Handle system time going backwards (Sometimes causing error)
            }
        } else {
            // If no levels exist, create one
            true
        }
    }

    fn create_new_level(&mut self) {
        let level = SlidingBloomFilterLevel {
            bit_vector: B::new(self.bit_vector_size),
            timestamp: SystemTime::now(),
        };
        self.levels.push(level);
        // Ensure we don't exceed the maximum number of levels
        if self.levels.len() > self.max_levels {
            self.levels.remove(0);
        }
    }

    pub fn insert(&mut self, item: &[u8]) {
        if self.should_create_new_level() {
            self.create_new_level();
        }
        let latest_level = self.levels.last_mut().unwrap();
        let hashes = (self.hash_function)(item, self.num_hashes, self.capacity);
        for hash in hashes {
            latest_level.bit_vector.set(hash as usize);
        }
    }

    pub fn query(&self, item: &[u8]) -> bool {
        for level in &self.levels {
            let hashes =
                (self.hash_function)(item, self.num_hashes, self.capacity);
            if hashes
                .iter()
                .all(|&hash| level.bit_vector.get(hash as usize))
            {
                return true;
            }
        }
        false
    }
}

impl<B: BitVector> std::fmt::Debug for SlidingBloomFilter<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SlidingBloomFilter {{ capacity: {}, num_hashes: {}, false_positive_rate: {}, level_time: {:?}, max_levels: {} }}",
            self.capacity,
            self.num_hashes,
            self.false_positive_rate,
            self.level_time,
            self.max_levels
        )
    }
}

pub struct BitVecBitVector {
    bits: BitVec,
}

impl BitVector for BitVecBitVector {
    fn new(size: usize) -> Self {
        BitVecBitVector {
            bits: bitvec![0; size],
        }
    }

    fn set(&mut self, index: usize) {
        self.bits.set(index, true);
    }

    fn get(&self, index: usize) -> bool {
        self.bits[index]
    }

    fn clear(&mut self) {
        self.bits.fill(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::thread;

    #[test]
    fn test_workflow() {
        let hash_function = |item: &[u8],
                             num_hashes_var: usize,
                             capacity_var: usize|
         -> Vec<u32> {
            let h1 = hash_murmur32(item);
            let h2 = hash_fnv32(item);
            (0..num_hashes_var)
                .map(|i| {
                    h1.wrapping_add((i as u32).wrapping_mul(h2))
                        % capacity_var as u32
                })
                .collect()
        };
        const capacity: usize = 1000;
        const num_hashes: usize = 7;
        const false_positive_rate: f64 = 0.01;
        const decay_time: Duration = Duration::new(10, 0);

        let mut bloom_filter = SlidingBloomFilter::<BitVecBitVector>::new(
            capacity,
            false_positive_rate,
            decay_time,
            num_hashes,
            hash_function,
        );
        bloom_filter.insert(b"some data");
        bloom_filter.insert(b"another data");
        assert!(bloom_filter.query(b"some data"));
        assert!(bloom_filter.query(b"another data"));
        assert!(!bloom_filter.query(b"some"));
        assert!(!bloom_filter.query(b"another"));
    }

    #[test]
    fn test_expiration_of_elements() {
        let mut bloom_filter = SlidingBloomFilter::<BitVecBitVector>::new(
            100,
            0.01,
            Duration::from_secs(1),
            2,
            default_hash_function,
        );

        bloom_filter.insert(b"item1");
        assert!(bloom_filter.query(b"item1"));

        // Wait enough time for the item to expire
        thread::sleep(Duration::from_secs(5)); // Exceeds MAX_LEVELS * LEVEL_TIME

        // Call cleanup explicitly
        bloom_filter.cleanup_expired_levels();

        assert!(!bloom_filter.query(b"item1"));
    }

    #[test]
    fn test_no_false_negatives_within_decay_time() {
        let mut bloom_filter = SlidingBloomFilter::<BitVecBitVector>::new(
            1000,
            0.01,
            Duration::from_secs(2),
            5,
            default_hash_function,
        );

        let items: Vec<&[u8]> =
            vec![b"apple", b"banana", b"cherry", b"date", b"elderberry"];

        for item in &items {
            bloom_filter.insert(item);
        }

        // Query immediately
        for item in &items {
            assert!(bloom_filter.query(item));
        }

        // Wait less than total decay time
        thread::sleep(Duration::from_secs(5)); // Less than MAX_LEVELS * LEVEL_TIME
        bloom_filter.cleanup_expired_levels();

        for item in &items {
            assert!(bloom_filter.query(item));
        }
    }

    #[test]
    fn test_items_expire_after_decay_time() {
        let mut bloom_filter = SlidingBloomFilter::<BitVecBitVector>::new(
            100,
            0.01,
            Duration::from_secs(1),
            3,
            default_hash_function,
        );

        bloom_filter.insert(b"item_to_expire");
        assert!(bloom_filter.query(b"item_to_expire"));

        // Wait for the item to expire
        thread::sleep(Duration::from_secs(4)); // Exceeds MAX_LEVELS * LEVEL_TIME
        bloom_filter.cleanup_expired_levels();

        assert!(!bloom_filter.query(b"item_to_expire"));
    }

    #[test]
    fn test_continuous_insertion_and_query() {
        let mut bloom_filter = SlidingBloomFilter::<BitVecBitVector>::new(
            1000,
            0.01,
            Duration::from_secs(1),
            5,
            default_hash_function,
        );

        for i in 0..10 {
            let item = format!("item_{}", i);
            bloom_filter.insert(item.as_bytes());
            assert!(bloom_filter.query(item.as_bytes()));
            thread::sleep(Duration::from_millis(400));
        }

        // Wait for earlier items to expire
        thread::sleep(Duration::from_secs(3));
        bloom_filter.cleanup_expired_levels();

        // Items 0 to 4 should have expired
        for i in 0..6 {
            let item = format!("item_{}", i);
            assert!(
                !bloom_filter.query(item.as_bytes()),
                "Item {} should have expired",
                item
            );
        }

        // Items 5 to 9 should still be present
        for i in 6..10 {
            let item = format!("item_{}", i);
            assert!(
                bloom_filter.query(item.as_bytes()),
                "Item {} should still be present",
                item
            );
        }
    }

    #[test]
    fn test_false_positive_rate() {
        const FALSE_POSITIVE_RATE: f64 = 0.05;

        let mut bloom_filter = SlidingBloomFilter::<BitVecBitVector>::new(
            10000,
            FALSE_POSITIVE_RATE,
            Duration::from_secs(2),
            5,
            default_hash_function,
        );

        let num_items = 1000;
        let mut rng = rand::thread_rng();
        let mut inserted_items = Vec::new();

        // Insert random items
        for _ in 0..num_items {
            let item: Vec<u8> = (0..10).map(|_| rng.gen()).collect();
            bloom_filter.insert(&item);
            inserted_items.push(item);
        }

        // Test for false positives
        let mut false_positives = 0;
        let num_tests = 1000;

        bloom_filter.cleanup_expired_levels();

        for _ in 0..num_tests {
            let item: Vec<u8> = (0..10).map(|_| rng.gen()).collect();
            if bloom_filter.query(&item) {
                // Check if the item was actually inserted
                if !inserted_items.contains(&item) {
                    false_positives += 1;
                }
            }
        }

        let observed_fpr = false_positives as f64 / num_tests as f64;
        assert!(
            observed_fpr <= FALSE_POSITIVE_RATE * 1.5,
            "False positive rate is too high: observed {}, expected {}",
            observed_fpr,
            FALSE_POSITIVE_RATE
        );
    }

    #[test]
    fn test_concurrent_inserts() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let bloom_filter =
            Arc::new(Mutex::new(SlidingBloomFilter::<BitVecBitVector>::new(
                1000,
                0.01,
                Duration::from_secs(1),
                5,
                default_hash_function,
            )));

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let bloom_filter = Arc::clone(&bloom_filter);
                thread::spawn(move || {
                    let item = format!("concurrent_item_{}", i);
                    let mut bf = bloom_filter.lock().unwrap();
                    bf.insert(item.as_bytes());
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        bloom_filter.lock().unwrap().cleanup_expired_levels();

        // Verify that all items have been inserted
        for i in 0..10 {
            let item = format!("concurrent_item_{}", i);
            let bf = bloom_filter.lock().unwrap();
            assert!(bf.query(item.as_bytes()));
        }
    }

    #[test]
    fn test_full_capacity() {
        const FALSE_POSITIVE_RATE: f64 = 0.1;

        let mut bloom_filter = SlidingBloomFilter::<BitVecBitVector>::new(
            100,
            FALSE_POSITIVE_RATE,
            Duration::from_secs(1),
            5,
            default_hash_function,
        );

        // Insert more items than capacity to test behavior
        for i in 0..200 {
            let item = format!("item_{}", i);
            bloom_filter.insert(item.as_bytes());

            bloom_filter.cleanup_expired_levels();
            assert!(bloom_filter.query(item.as_bytes()));
        }

        bloom_filter.cleanup_expired_levels();
        // Expect higher false positive rate due to saturation
        let false_queries = (200..300)
            .filter(|i| {
                let item = format!("item_{}", i);
                bloom_filter.query(item.as_bytes())
            })
            .count();

        let observed_fpr = false_queries as f64 / 100.0;
        assert!(
            observed_fpr >= FALSE_POSITIVE_RATE,
            "False positive rate is lower than expected: observed {}, expected {}",
            observed_fpr,
            FALSE_POSITIVE_RATE
        );
    }

    #[test]
    fn test_clear_functionality() {
        let mut bloom_filter = SlidingBloomFilter::<BitVecBitVector>::new(
            1000,
            0.01,
            Duration::from_secs(1),
            5,
            default_hash_function,
        );

        bloom_filter.insert(b"persistent_item");

        // Insert items that should expire
        bloom_filter.insert(b"temp_item");

        bloom_filter.cleanup_expired_levels();
        assert!(bloom_filter.query(b"temp_item"));

        // Wait for the temporary item to expire
        thread::sleep(Duration::from_secs(6)); // Exceeds MAX_LEVELS * LEVEL_TIME
        bloom_filter.cleanup_expired_levels();

        // "temp_item" should be expired
        assert!(!bloom_filter.query(b"temp_item"));

        // "persistent_item" should be also expired
        assert!(!bloom_filter.query(b"persistent_item"));
    }

    #[test]
    fn test_should_create_new_level_edge_case() {
        const MAX_LEVELS: usize = 3;

        let mut bloom_filter = SlidingBloomFilter::<BitVecBitVector>::new(
            1000,
            0.01,
            Duration::from_millis(500),
            MAX_LEVELS,
            default_hash_function,
        );

        // Rapid insertions to test level creation
        for i in 0..10 {
            let item = format!("rapid_item_{}", i);
            bloom_filter.insert(item.as_bytes());
            thread::sleep(Duration::from_millis(100)); // Sleep less than LEVEL_TIME
        }

        // Levels should have been created appropriately
        assert!(bloom_filter.levels.len() <= MAX_LEVELS);
    }
}
