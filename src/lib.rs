use bitline::BitLine;
use bytes::Bytes;
use cloud_storage::Object;
use futures_util::stream::{StreamExt};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::{basic::CompareOp, types::PyBytes, types::PyTuple};
use std::fs::File;
use std::io::{Read, Write};
use std::mem;
use std::path::PathBuf;
use tokio::runtime::Runtime;

#[pyclass(module = "rbloom")]
#[derive(Clone)]
struct Bloom {
    filter: BitLine,
    k: u64, // Number of hash functions
}

#[pymethods]
impl Bloom {
    #[new]
    fn new(
        expected_items: u64,
        false_positive_rate: f64,
    ) -> PyResult<Self> {
        // Check the inputs
        if false_positive_rate <= 0.0 || false_positive_rate >= 1.0 {
            return Err(PyValueError::new_err(
                "false_positive_rate must be between 0 and 1",
            ));
        }
        if expected_items == 0 {
            return Err(PyValueError::new_err(
                "expected_items must be greater than 0",
            ));
        }

        // Calculate the parameters for the filter
        let size_in_bits =
            -1.0 * (expected_items as f64) * false_positive_rate.ln() / 2.0f64.ln().powi(2);
        let k = (size_in_bits / expected_items as f64) * 2.0f64.ln();

        // Create the filter
        Ok(Bloom {
            filter: BitLine::new(size_in_bits as u64)?,
            k: k as u64,
        })
    }

    /// Number of buckets in the filter
    #[getter]
    fn size_in_bits(&self) -> u64 {
        self.filter.len()
    }

    /// Estimated number of items in the filter
    #[getter]
    fn approx_items(&self) -> f64 {
        let len = self.filter.len() as f64;
        let bits_set = self.filter.sum() as f64;
        (len / (self.k as f64) * (1.0 - (bits_set) / len).ln()).abs()
    }

    #[pyo3(signature = (hashed, /))]
    fn add(&mut self, hashed: i128) -> PyResult<()> {
        for index in lcg::generate_indexes(hashed, self.k, self.filter.len()) {
            self.filter.set(index);
        }
        Ok(())
    }

    /// Test whether every element in the bloom may be in other
    ///
    /// This can have false positives (return true for a bloom which does not
    /// contain all items in this set), but it will not return a false negative:
    /// If this returns false, this set contains an element which is not in other
    #[pyo3(signature = (other, /))]
    fn issubset(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.with_other_as_bloom(other, |other_bloom| {
            Ok(self.filter.is_subset(&other_bloom.filter))
        })
    }

    /// Test whether every element in other may be in self
    ///
    /// This can have false positives (return true for a bloom which does not
    /// contain all items in other), but it will not return a false negative:
    /// If this returns false, other contains an element which is not in self
    #[pyo3(signature = (other, /))]
    fn issuperset(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.with_other_as_bloom(other, |other_bloom| {
            Ok(other_bloom.filter.is_subset(&self.filter))
        })
    }


    fn __contains__(&self, hashed: i128) -> PyResult<bool> {
        for index in lcg::generate_indexes(hashed, self.k, self.filter.len()) {
            if !self.filter.get(index) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Return a new set with elements from the set and all others.
    #[pyo3(signature = (*others))]
    fn union(&self, others: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let mut result = self.clone();
        result.update(others)?;
        Ok(result)
    }

    /// Return a new set with elements common to the set and all others.
    #[pyo3(signature = (*others))]
    fn intersection(&self, others: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let mut result = self.clone();
        result.intersection_update(others)?;
        Ok(result)
    }

    fn __or__(&self, _py: Python<'_>, other: &Bloom) -> PyResult<Bloom> {
        check_compatible(self, other)?;
        Ok(Bloom {
            filter: &self.filter | &other.filter,
            k: self.k,
        })
    }

    fn __ior__(&mut self, other: &Bloom) -> PyResult<()> {
        check_compatible(self, other)?;
        self.filter |= &other.filter;
        Ok(())
    }

    fn __and__(&self, _py: Python<'_>, other: &Bloom) -> PyResult<Bloom> {
        check_compatible(self, other)?;
        Ok(Bloom {
            filter: &self.filter & &other.filter,
            k: self.k,
        })
    }

    fn __iand__(&mut self, other: &Bloom) -> PyResult<()> {
        check_compatible(self, other)?;
        self.filter &= &other.filter;
        Ok(())
    }

    #[pyo3(signature = (*others))]
    fn update(&mut self, others: &Bound<'_, PyTuple>) -> PyResult<()> {
        for other in others.iter() {
            // If the other object is a Bloom, use the bitwise union
            if let Ok(other) = other.downcast::<Bloom>() {
                let other = other.try_borrow()?;
                self.__ior__(&other)?;
            }
            // Otherwise, iterate over the other object and add each item
            else {
                for obj in other.try_iter()? {
                    let h: i128 = obj?.extract()?;
                    self.add(h)?;
                }
            }
        }
        Ok(())
    }

    #[pyo3(signature = (*others))]
    fn intersection_update(&mut self, others: &Bound<'_, PyTuple>) -> PyResult<()> {
        // Lazily allocated temp bitset
        let mut temp: Option<Self> = None;
        for other in others.iter() {
            // If the other object is a Bloom, use the bitwise intersection
            if let Ok(other) = other.downcast::<Bloom>() {
                let other = other.try_borrow()?;
                self.__iand__(&other)?;
            }
            // Otherwise, iterate over the other object and add each item
            else {
                let temp = temp.get_or_insert_with(|| self.clone());
                temp.clear();
                for obj in other.try_iter()? {
                    let h: i128 = obj?.extract()?;
                    temp.add(h)?;
                }
                self.__iand__(temp)?;
            }
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.filter.clear();
    }

    fn copy(&self) -> Bloom {
        self.clone()
    }

    fn __repr__(&self) -> String {
        // Use a format that makes it clear that the object
        // cannot be reconstructed from the repr
        format!(
            "<Bloom size_in_bits={} approx_items={:.1}>",
            self.size_in_bits(),
            self.approx_items()
        )
    }

    fn __bool__(&self) -> bool {
        !self.filter.is_empty()
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        check_compatible(self, other)?;
        Ok(match op {
            CompareOp::Eq => self.filter == other.filter,
            CompareOp::Ne => self.filter != other.filter,
            CompareOp::Le => self.filter.is_subset(&other.filter),
            CompareOp::Lt => self.filter.is_strict_subset(&other.filter),
            CompareOp::Ge => other.filter.is_subset(&self.filter),
            CompareOp::Gt => other.filter.is_strict_subset(&self.filter),
        })
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    /// Load from a file, see "Persistence" section in the README
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filepath: PathBuf,
    ) -> PyResult<Bloom> {
        let mut file = File::open(filepath)?;

        let mut k_bytes = [0; mem::size_of::<u64>()];
        file.read_exact(&mut k_bytes)?;
        let k = u64::from_le_bytes(k_bytes);

        let filter = BitLine::load(&mut file)?;

        Ok(Bloom {
            filter,
            k,
        })
    }

    /// Load from a bytes(), see "Persistence" section in the README
    #[classmethod]
    fn load_bytes(
        _cls: &Bound<'_, PyType>,
        bytes: &[u8],
    ) -> PyResult<Bloom> {
        let k_bytes: [u8; mem::size_of::<u64>()] = bytes[0..mem::size_of::<u64>()]
            .try_into()
            .expect("slice with incorrect length");
        let k = u64::from_le_bytes(k_bytes);

        let filter = BitLine::load_bytes(&bytes[mem::size_of::<u64>()..])?;

        Ok(Bloom {
            filter,
            k,
        })
    }

    /// Load a Bloom filter from Google Cloud Storage
    /// without holding two copies of the bit array in memory.
    #[classmethod]
    pub fn load_from_gcs(_cls: &Bound<'_, PyType>, bucket: String, object_path: String) -> PyResult<Bloom> {
        // 1) Download the entire object (synchronous) into *one* Vec<u8>.
        let mut data = Object::download_sync(&bucket, &object_path)
            .map_err(|e| PyValueError::new_err(format!("GCS download failed: {e}")))?;

        // 2) Parse the first 8 bytes to get `k`.
        if data.len() < mem::size_of::<u64>() {
            return Err(PyValueError::new_err(
                "Object is too small to contain a valid Bloom filter",
            ));
        }
        let (k_bytes, _) = data.split_at(mem::size_of::<u64>());
        let k = u64::from_le_bytes(k_bytes.try_into().unwrap());

        // 3) Remove those 8 bytes from the front of `data`.
        // Now `data` is exactly the raw bits used by BitLine.
        // Draining 0..8 does *not* copy the remainder.
        data.drain(0..mem::size_of::<u64>());

        // 4) Build the BitLine directly from this vector.
        //    `Vec::into_boxed_slice()` does not copy on stable Rust >= 1.48.
        let filter = BitLine::from_boxed_slice(data.into_boxed_slice());

        Ok(Bloom { filter, k })
    }

    /// Load a Bloom filter from Google Cloud Storage by streaming in larger
    /// chunks. This function uses `Object::download_bytes_stream` from
    /// https://github.com/ThouCheese/cloud-storage-rs/pull/123, which yields
    /// `Bytes` rather than one byte per `Stream` item (as in
    /// `Object::download_streamed`), significantly reducing overhead.
    #[classmethod]
    #[pyo3(signature = (
        bucket,
        object_path,
    ))]
    pub fn load_from_gcs_streamed(
        _cls: &Bound<'_, PyType>,
        bucket: String,
        object_path: String,
    ) -> PyResult<Bloom> {
        // Create a local Tokio runtime.
        let rt = Runtime::new()
            .map_err(|e| PyValueError::new_err(format!("Failed to create Tokio runtime: {e}")))?;

        rt.block_on(async move {
            // Open a chunked download stream. Each `next()` yields `Bytes`.
            let mut stream = Object::download_bytes_stream(&bucket, &object_path)
                .await
                .map_err(|e| PyValueError::new_err(format!("GCS download failed: {e}")))?;

            // We need the first 8 bytes to parse `k`.
            // `leftover` will accumulate data until we have at least 8 bytes.
            let mut leftover: Vec<u8> = Vec::with_capacity(8);

            // Fill `leftover` until we have at least 8 bytes for `k`.
            while leftover.len() < 8 {
                let next_chunk = match stream.next().await {
                    Some(Ok(chunk)) => chunk,
                    Some(Err(e)) => {
                        return Err(PyValueError::new_err(format!(
                            "Stream read error while reading k: {e}"
                        )));
                    }
                    None => {
                        return Err(PyValueError::new_err(
                            "Object is too small to contain a valid Bloom filter",
                        ));
                    }
                };
                leftover.extend_from_slice(&next_chunk);
            }

            // 4) Now parse the first 8 bytes as `k`.
            let k_bytes: [u8; 8] = leftover[..8].try_into().unwrap();
            let k = u64::from_le_bytes(k_bytes);

            // The remainder of `leftover` (if any) is the first part of the filter bits.
            let mut bit_buffer = leftover.split_off(8);

            // Consume the rest of the chunked stream into `bit_buffer`.
            while let Some(chunk_result) = stream.next().await {
                let chunk: Bytes = chunk_result.map_err(|e| {
                    PyValueError::new_err(format!("Stream read error in chunk: {e}"))
                })?;
                bit_buffer.extend_from_slice(&chunk);
            }

            // Convert our collected data to a BitLine.
            let filter = BitLine::from_boxed_slice(bit_buffer.into_boxed_slice());

            // Return the Bloom filter object.
            Ok(Bloom { filter, k })
        })
    }
    /// Save to a file, see "Persistence" section in the README
    fn save(&self, filepath: PathBuf) -> PyResult<()> {
        let mut file = File::create(filepath)?;
        file.write_all(&self.k.to_le_bytes())?;
        self.filter.save(&mut file)?;
        Ok(())
    }

    /// Save to a byte(), see "Persistence" section in the README
    fn save_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        const K_SIZE: usize = mem::size_of::<u64>();
        debug_assert_eq!(K_SIZE, self.k.to_le_bytes().len());
        let len = K_SIZE + self.filter.bits().len();
        PyBytes::new_with(py, len, |data| {
            data[..K_SIZE].copy_from_slice(&self.k.to_le_bytes());
            data[K_SIZE..].copy_from_slice(self.filter.bits());
            Ok(())
        })
    }
}

// Non-python methods
impl Bloom {
    fn zeroed_clone(&self, _py: Python<'_>) -> Bloom {
        Bloom {
            filter: BitLine::new(self.filter.len()).unwrap(),
            k: self.k,
        }
    }

    /// Extract other as a bloom, or iterate other, and add all items to a temporary bloom
    fn with_other_as_bloom<O>(
        &self,
        other: &Bound<'_, PyAny>,
        f: impl FnOnce(&Bloom) -> PyResult<O>,
    ) -> PyResult<O> {
        match other.downcast::<Bloom>() {
            Ok(o) => {
                let o = o.try_borrow()?;
                check_compatible(self, &o)?;
                f(&o)
            }
            Err(_) => {
                let mut other_bloom = self.zeroed_clone(other.py());
                for obj in other.try_iter()? {
                    let h: i128 = obj?.extract()?;
                    other_bloom.add(h)?;
                }
                f(&other_bloom)
            }
        }
    }
}

/// This is a primitive BitVec-like structure that uses a `Box<[u8]>` as
/// the backing store; it exists here to avoid the need for a dependency
/// on bitvec and to act as a container around all the bit manipulation.
/// Indexing is done using u64 to avoid address space issues on 32-bit
/// systems, which would otherwise limit the size to 2^32 bits (512MB).
/// Using u8 for the backing store simplifies file I/O as well as file
/// portability across systems, and the performance is equivalent to
/// using usize, even though the latter is arguably more elegant.
mod bitline {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use std::fs::File;
    use std::io::{Read, Write};

    #[inline(always)]
    fn bit_idx(idx: u64) -> Option<(usize, u32)> {
        let (q, r) = (idx / 8, idx % 8);
        Some((q.try_into().ok()?, r.try_into().ok()?))
    }

    #[derive(Clone, PartialEq, Eq)]
    pub struct BitLine {
        bits: Box<[u8]>,
    }

    impl BitLine {
        pub fn new(size_in_bits: u64) -> PyResult<Self> {
            match bit_idx(size_in_bits) {
                Some((q, r)) => {
                    let size = if r == 0 { q } else { q + 1 };
                    Ok(Self {
                        bits: vec![0; size].into_boxed_slice(),
                    })
                }
                None => Err(PyValueError::new_err("too many bits")),
            }
        }

        /// Create a BitLine directly from a boxed slice
        pub fn from_boxed_slice(bits: Box<[u8]>) -> Self {
            BitLine { bits }
        }

        /// Make sure that index is less than len when calling this!
        pub fn set(&mut self, index: u64) {
            let (idx, offset) = bit_idx(index).unwrap();
            self.bits[idx] |= 1 << offset;
        }

        /// Make sure that index is less than len when calling this!
        pub fn get(&self, index: u64) -> bool {
            let (idx, offset) = bit_idx(index).unwrap();
            self.bits[idx] & (1 << offset) != 0
        }

        /// Returns the number of bits in the BitLine
        pub fn len(&self) -> u64 {
            self.bits.len() as u64 * 8
        }

        pub fn clear(&mut self) {
            self.bits.fill(0);
        }

        pub fn sum(&self) -> u64 {
            self.bits.iter().map(|x| x.count_ones() as u64).sum()
        }

        pub fn is_empty(&self) -> bool {
            self.bits.iter().all(|&word| word == 0)
        }

        pub fn is_subset(&self, other: &BitLine) -> bool {
            all_pairs(self, other, |lhs, rhs| (lhs | rhs) == rhs)
        }

        pub fn is_strict_subset(&self, other: &BitLine) -> bool {
            let mut is_equal = true;
            let is_subset = all_pairs(self, other, |lhs, rhs| {
                is_equal &= lhs == rhs;
                (lhs | rhs) == rhs
            });
            is_subset && !is_equal
        }

        /// Reads the given file from the current position to the end and
        /// returns a BitLine containing the data.
        pub fn load(file: &mut File) -> PyResult<Self> {
            let mut bits = Vec::new();
            file.read_to_end(&mut bits)?;
            Ok(Self {
                bits: bits.into_boxed_slice(),
            })
        }

        /// Given the provided [u8]
        /// returns a BitLine containing the data.
        pub fn load_bytes(bytes: &[u8]) -> PyResult<Self> {
            let bits = bytes.to_vec();
            Ok(Self {
                bits: bits.into_boxed_slice(),
            })
        }

        /// Writes the BitLine to the given file from the current position.
        pub fn save(&self, file: &mut File) -> PyResult<()> {
            file.write_all(&self.bits)?;
            Ok(())
        }

        pub fn bits(&self) -> &[u8] {
            &self.bits
        }
    }

    fn all_pairs(lhs: &BitLine, rhs: &BitLine, mut f: impl FnMut(u8, u8) -> bool) -> bool {
        lhs.bits
            .iter()
            .zip(rhs.bits.iter())
            .all(move |(&lhs, &rhs)| f(lhs, rhs))
    }

    impl std::ops::BitAnd for BitLine {
        type Output = Self;

        fn bitand(mut self, rhs: Self) -> Self::Output {
            self &= rhs;
            self
        }
    }

    impl std::ops::BitAnd for &BitLine {
        type Output = BitLine;

        fn bitand(self, rhs: Self) -> Self::Output {
            let mut result = self.clone();
            result &= rhs;
            result
        }
    }

    impl std::ops::BitAndAssign for BitLine {
        fn bitand_assign(&mut self, rhs: Self) {
            *self &= &rhs;
        }
    }
    impl std::ops::BitAndAssign<&BitLine> for BitLine {
        fn bitand_assign(&mut self, rhs: &Self) {
            for (lhs, rhs) in self.bits.iter_mut().zip(rhs.bits.iter()) {
                *lhs &= rhs;
            }
        }
    }

    impl std::ops::BitOr for BitLine {
        type Output = Self;

        fn bitor(mut self, rhs: Self) -> Self::Output {
            self |= rhs;
            self
        }
    }

    impl std::ops::BitOr for &BitLine {
        type Output = BitLine;

        fn bitor(self, rhs: Self) -> Self::Output {
            let mut result = self.clone();
            result |= rhs;
            result
        }
    }

    impl std::ops::BitOrAssign for BitLine {
        fn bitor_assign(&mut self, rhs: Self) {
            *self |= &rhs;
        }
    }

    impl std::ops::BitOrAssign<&BitLine> for BitLine {
        fn bitor_assign(&mut self, rhs: &Self) {
            for (lhs, rhs) in self.bits.iter_mut().zip(rhs.bits.iter()) {
                *lhs |= rhs;
            }
        }
    }
}

/// This implements a linear congruential generator that is
/// used to distribute entropy from the hash over multiple ints.
mod lcg {
    pub struct Random {
        state: u128,
    }

    impl Iterator for Random {
        type Item = u64;

        fn next(&mut self) -> Option<Self::Item> {
            self.state = self
                .state
                .wrapping_mul(47026247687942121848144207491837418733)
                .wrapping_add(1);
            Some((self.state >> 32) as Self::Item)
        }
    }

    pub fn distribute_entropy(hash: i128) -> Random {
        Random {
            state: hash as u128,
        }
    }

    pub fn generate_indexes(hash: i128, k: u64, len: u64) -> impl Iterator<Item = u64> {
        distribute_entropy(hash)
            .take(k as usize)
            .map(move |x: u64| x % len)
    }
}


fn check_compatible(a: &Bloom, b: &Bloom) -> PyResult<()> {
    if a.k != b.k || a.filter.len() != b.filter.len() {
        return Err(PyValueError::new_err(
            "size and max false positive rate must be the same for both filters",
        ));
    }

    Ok(())
}


#[pymodule]
fn rbloom(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bloom>()?;
    Ok(())
}
