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

// Trait for hash functions
pub trait HashFunction {
    fn hash(&self, item: &[u8]) -> Vec<usize>;
    // fn hash(&self, item: &[u8], number_of_hashes: usize) -> Vec<usize>;
}

// Structure for a single Bloom filter level
pub struct BloomFilterLevel<B: BitVector> {
    bit_vector: B,
    timestamp: SystemTime,
}

// Main structure for the Time-Decaying Bloom Filter
pub struct TimeDecayingBloomFilter<B: BitVector, H: HashFunction> {
    levels: Vec<BloomFilterLevel<B>>,
    hash_functions: Vec<H>,
    capacity: usize,
    false_positive_rate: f64,
    decay_time: Duration,
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

// Trait for Time-Decaying Bloom Filter operations
pub trait BloomFilterOperations<B: BitVector, H: HashFunction> {
    fn new(
        capacity: usize,
        false_positive_rate: f64,
        decay_time: Duration,
        hash_functions: Vec<H>,
    ) -> Self;

    fn insert(&mut self, item: &[u8]);
    fn query(&self, item: &[u8]) -> bool;
    fn cleanup(&mut self);
}

// Implementation block (without actual implementations)
impl<B: BitVector, H: HashFunction> BloomFilterOperations<B, H>
    for TimeDecayingBloomFilter<B, H>
{
    fn new(
        capacity: usize,
        false_positive_rate: f64,
        decay_time: Duration,
        hash_functions: Vec<H>,
    ) -> Self {
        // Implementation here

        let num_bits = Self::calculate_num_bits(capacity, false_positive_rate);
        let level = BloomFilterLevel {
            bit_vector: B::new(num_bits),
            timestamp: SystemTime::now(),
        };

        TimeDecayingBloomFilter {
            levels: vec![level],
            hash_functions,
            capacity,
            false_positive_rate,
            decay_time,
        }
    }

    fn insert(&mut self, item: &[u8]) {
        let current_time = SystemTime::now();
        if self.levels.is_empty()
            || current_time
                .duration_since(self.levels.last().unwrap().timestamp)
                .unwrap()
                > self.decay_time
        {
            self.levels.push(BloomFilterLevel {
                bit_vector: B::new(Self::calculate_num_bits(
                    self.capacity,
                    self.false_positive_rate,
                )),
                timestamp: current_time,
            });
        }

        let level = self.levels.last_mut().unwrap();
        for hash_function in &self.hash_functions {
            for index in hash_function.hash(item) {
                level.bit_vector.set(
                    index
                        % Self::calculate_num_bits(
                            self.capacity,
                            self.false_positive_rate,
                        ),
                );
            }
        }
    }

    fn query(&self, item: &[u8]) -> bool {
        let current_time = SystemTime::now();
        self.levels
            .iter()
            .rev()
            .take_while(|level| {
                current_time.duration_since(level.timestamp).unwrap()
                    <= self.decay_time
            })
            .any(|level| {
                self.hash_functions.iter().all(|hash_function| {
                    hash_function.hash(item).iter().all(|&index| {
                        level.bit_vector.get(
                            index
                                % Self::calculate_num_bits(
                                    self.capacity,
                                    self.false_positive_rate,
                                ),
                        )
                    })
                })
            })
    }

    fn cleanup(&mut self) {
        let current_time = SystemTime::now();
        self.levels.retain(|level| {
            current_time.duration_since(level.timestamp).unwrap()
                <= self.decay_time
        });
    }
}

impl<B: BitVector, H: HashFunction> TimeDecayingBloomFilter<B, H> {
    fn calculate_num_bits(capacity: usize, false_positive_rate: f64) -> usize {
        let m =
            -(capacity as f64 * false_positive_rate.ln()) / (2f64.ln().powi(2));
        m.ceil() as usize
    }
}

pub struct MurmurFNVHashFunction {
    num_hashes: usize,
    bit_vector_size: usize,
}

impl MurmurFNVHashFunction {
    pub fn new(num_hashes: usize, bit_vector_size: usize) -> Self {
        MurmurFNVHashFunction {
            num_hashes,
            bit_vector_size,
        }
    }

    fn hash1(&self, key: &[u8]) -> u32 {
        let mut cursor = Cursor::new(key);
        murmur3_32(&mut cursor, 0).expect("Failed to compute Murmur3 hash")
    }

    fn hash2(&self, key: &[u8]) -> u32 {
        let mut hasher = FnvHasher::default();
        hasher.write(key);
        hasher.finish() as u32
    }
}

impl HashFunction for MurmurFNVHashFunction {
    fn hash(&self, item: &[u8]) -> Vec<usize> {
        let h1 = self.hash1(item);
        let h2 = self.hash2(item);

        (0..self.num_hashes)
            .map(|i| {
                ((h1 as u64 + i as u64 * h2 as u64) % self.bit_vector_size as u64)
                    as usize
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::thread;

    // Mock implementations for testing
    struct MockBitVector {
        bits: Vec<bool>,
    }

    impl BitVector for MockBitVector {
        fn new(size: usize) -> Self {
            MockBitVector {
                bits: vec![false; size],
            }
        }
        fn set(&mut self, index: usize) {
            self.bits[index] = true;
        }
        fn get(&self, index: usize) -> bool {
            self.bits[index]
        }
        fn clear(&mut self) {
            self.bits.fill(false);
        }
    }

    struct MockHashFunction;

    impl HashFunction for MockHashFunction {
        fn hash(&self, item: &[u8]) -> Vec<usize> {
            vec![item[0] as usize % 10, item[0] as usize % 20]
        }
    }

    #[test]
    fn test_correctness() {
        let hash_function1 = MurmurFNVHashFunction::new(3, 1000);
        let hash_function2 = MurmurFNVHashFunction::new(3, 1000);
        let mut bf = TimeDecayingBloomFilter::<
            BitVecBitVector,
            MurmurFNVHashFunction,
        >::new(
            1000,
            0.01,
            Duration::from_secs(1),
            vec![hash_function1, hash_function2],
        );

        // Test insertion and querying
        bf.insert(b"test_item");
        assert!(
            bf.query(b"test_item"),
            "Item should be found after insertion"
        );
        assert!(
            !bf.query(b"not_inserted"),
            "Non-inserted item should not be found"
        );

        // Test decay
        std::thread::sleep(Duration::from_secs(2));
        bf.cleanup();
        assert!(
            !bf.query(b"test_item"),
            "Item should not be found after decay time"
        );

        // Test multiple insertions
        for i in 0..100 {
            bf.insert(format!("item_{}", i).as_bytes());
        }

        // Verify all inserted items are found
        for i in 0..100 {
            assert!(
                bf.query(format!("item_{}", i).as_bytes()),
                "Item {} should be found",
                i
            );
        }

        // Verify false positive rate
        let mut false_positives = 0;
        for i in 100..1100 {
            // Test 1000 non-inserted items
            if bf.query(format!("non_item_{}", i).as_bytes()) {
                false_positives += 1;
            }
        }

        let observed_false_positive_rate = false_positives as f64 / 1000.0;
        println!(
            "Observed false positive rate: {}",
            observed_false_positive_rate
        );
        assert!(
            observed_false_positive_rate < 0.02,
            "False positive rate too high"
        );
    }

    #[test]
    fn test_new() {
        let bv = BitVecBitVector::new(100);
        assert_eq!(bv.bits.len(), 100);
        assert!(bv.bits.not_any());
    }

    #[test]
    fn test_set_and_get() {
        let mut bv = BitVecBitVector::new(100);
        bv.set(50);
        assert!(bv.get(50));
        assert!(!bv.get(49));
        assert!(!bv.get(51));
    }

    #[test]
    fn test_clear() {
        let mut bv = BitVecBitVector::new(100);
        bv.set(25);
        bv.set(75);
        assert!(bv.get(25));
        assert!(bv.get(75));
        bv.clear();
        assert!(!bv.get(25));
        assert!(!bv.get(75));
        assert!(bv.bits.not_any());
    }

    #[test]
    fn test_multiple_operations() {
        let mut bv = BitVecBitVector::new(1000);
        for i in (0..1000).step_by(2) {
            bv.set(i);
        }
        for i in 0..1000 {
            assert_eq!(bv.get(i), i % 2 == 0);
        }
        bv.clear();
        assert!(bv.bits.not_any());
    }

    #[test]
    fn test_new_bloom_filter() {
        let bf = TimeDecayingBloomFilter::<MockBitVector, MockHashFunction>::new(
            1000,
            0.01,
            Duration::from_secs(3600),
            vec![MockHashFunction, MockHashFunction],
        );

        assert_eq!(bf.capacity, 1000);
        assert_eq!(bf.false_positive_rate, 0.01);
        assert_eq!(bf.decay_time, Duration::from_secs(3600));
        assert_eq!(bf.hash_functions.len(), 2);
        assert_eq!(bf.levels.len(), 1); // Should start with one level
    }

    #[test]
    fn test_insert_and_query() {
        let mut bf =
            TimeDecayingBloomFilter::<MockBitVector, MockHashFunction>::new(
                1000,
                0.01,
                Duration::from_secs(3600),
                vec![MockHashFunction],
            );

        bf.insert(b"test");
        assert!(bf.query(b"test"), "Item should be found after insertion");
        assert!(
            !bf.query(b"not_inserted"),
            "Non-inserted item should not be found"
        );
    }

    #[test]
    fn test_decay() {
        let mut bf =
            TimeDecayingBloomFilter::<MockBitVector, MockHashFunction>::new(
                1000,
                0.01,
                Duration::from_secs(1), // 1 second decay time for testing
                vec![MockHashFunction],
            );

        bf.insert(b"test");
        assert!(
            bf.query(b"test"),
            "Item should be found immediately after insertion"
        );

        // Simulate passage of time
        std::thread::sleep(Duration::from_secs(2));

        bf.cleanup();
        assert!(
            !bf.query(b"test"),
            "Item should not be found after decay time"
        );
    }

    #[test]
    fn test_murmur_fnv_hash_function() {
        let hash_function = MurmurFNVHashFunction::new(5, 100);
        let key = b"test_key";
        let hashes = hash_function.hash(key);

        assert_eq!(hashes.len(), 5);
        assert!(hashes.iter().all(|&h| h < 100));

        // Check that hashes are different
        assert!(hashes.windows(2).all(|w| w[0] != w[1]));
    }

    #[test]
    fn test_consistency() {
        let hash_function = MurmurFNVHashFunction::new(3, 1000);
        let key = b"another_test_key";
        let hashes1 = hash_function.hash(key);
        let hashes2 = hash_function.hash(key);

        assert_eq!(hashes1, hashes2, "Hash function should be deterministic");
    }

    #[test]
    fn test_murmur_fnv_hash_function_creation() {
        let hash_function = MurmurFNVHashFunction::new(5, 100);
        assert_eq!(hash_function.num_hashes, 5);
        assert_eq!(hash_function.bit_vector_size, 100);
    }

    #[test]
    fn test_hash_output_length() {
        let hash_function = MurmurFNVHashFunction::new(5, 100);
        let key = b"test_key";
        let hashes = hash_function.hash(key);
        assert_eq!(hashes.len(), 5);
    }

    #[test]
    fn test_hash_output_range() {
        let bit_vector_size = 100;
        let hash_function = MurmurFNVHashFunction::new(5, bit_vector_size);
        let key = b"test_key";
        let hashes = hash_function.hash(key);
        assert!(hashes.iter().all(|&h| h < bit_vector_size));
    }

    #[test]
    fn test_hash_consistency() {
        let hash_function = MurmurFNVHashFunction::new(5, 100);
        let key = b"test_key";
        let hashes1 = hash_function.hash(key);
        let hashes2 = hash_function.hash(key);
        assert_eq!(hashes1, hashes2, "Hash function should be deterministic");
    }

    #[test]
    fn test_hash_uniqueness() {
        let hash_function = MurmurFNVHashFunction::new(5, 1000);
        let key1 = b"test_key_1";
        let key2 = b"test_key_2";
        let hashes1 = hash_function.hash(key1);
        let hashes2 = hash_function.hash(key2);
        assert_ne!(
            hashes1, hashes2,
            "Different keys should produce different hash sets"
        );
    }

    #[test]
    fn test_hash_distribution() {
        let bit_vector_size = 1000;
        let hash_function = MurmurFNVHashFunction::new(5, bit_vector_size);
        let mut hash_counts = vec![0; bit_vector_size];

        // Generate hashes for a large number of keys
        for i in 0..10000 {
            let key = format!("test_key_{}", i).into_bytes();
            let hashes = hash_function.hash(&key);
            for hash in hashes {
                hash_counts[hash] += 1;
            }
        }

        // Check that all positions have been hashed to at least once
        assert!(hash_counts.iter().all(|&count| count > 0));

        // Check that no position has been hashed to excessively often
        let max_count = *hash_counts.iter().max().unwrap();
        let min_count = *hash_counts.iter().min().unwrap();
        println!("Max count: {}, Min count: {}", max_count, min_count);
        assert!(
            (max_count as f64 / min_count as f64) < 3.0,
            "Hash distribution is too uneven"
        );
    }

    #[test]
    fn test_multiple_levels() {
        let mut bf =
            TimeDecayingBloomFilter::<MockBitVector, MockHashFunction>::new(
                1000,
                0.01,
                Duration::from_millis(5),
                vec![MockHashFunction],
            );

        bf.insert(b"level1");
        thread::sleep(Duration::from_millis(10));
        bf.insert(b"level2");

        assert_eq!(
            bf.levels.len(),
            2,
            "Should have two levels after inserting with delay"
        );
        assert!(bf.query(b"level1"));
        assert!(bf.query(b"level2"));
    }

    #[test]
    fn test_hash_collision_resistance() {
        let hash_function = MurmurFNVHashFunction::new(5, 1000);
        let mut unique_hashes = HashSet::new();

        // Generate hashes for a large number of keys
        for i in 0..10000 {
            let key = format!("test_key_{}", i).into_bytes();
            let hashes = hash_function.hash(&key);
            unique_hashes.insert(hashes);
        }

        // Check that we have a high number of unique hash sets
        assert!(unique_hashes.len() > 9900, "Too many hash collisions");
    }

    #[test]
    fn test_different_num_hashes() {
        let hash_function1 = MurmurFNVHashFunction::new(3, 100);
        let hash_function2 = MurmurFNVHashFunction::new(5, 100);
        let key = b"test_key";
        let hashes1 = hash_function1.hash(key);
        let hashes2 = hash_function2.hash(key);
        assert_eq!(hashes1.len(), 3);
        assert_eq!(hashes2.len(), 5);
        assert_eq!(hashes1, &hashes2[0..3]);
    }
}
