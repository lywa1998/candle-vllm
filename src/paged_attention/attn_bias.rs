use std::iter::zip;

use candle_core::{DType, Device, Shape, Tensor};

use crate::openai::responses::APIError;

pub trait AttentionBias {
    fn materialize(&self, shape: &Shape, dtype: DType, device: Device) -> Result<Tensor, APIError>;
    fn from_seqlens(
        &self,
        q_seqlen: Vec<u32>,
        kv_seqlen: Option<Vec<u32>>,
        dtype: DType,
        device: Device,
    ) -> Result<Box<dyn AttentionBias>, APIError> {
        unimplemented!()
    }
}

struct SeqLenInfo {
    seqstart: Tensor,
    max_seqlen: u32,
    min_seqlen: u32,
    seqstart_py: Vec<u32>,
}

impl SeqLenInfo {
    fn new(seqstart: Tensor, max_seqlen: u32, min_seqlen: u32, seqstart_py: Vec<u32>) -> Self {
        Self {
            seqstart,
            max_seqlen,
            min_seqlen,
            seqstart_py,
        }
    }

    fn from_seqlens(
        seqlens: impl Iterator<Item = u32>,
        dtype: DType,
        device: Device,
    ) -> Result<Self, APIError> {
        let mut seqstart_py = vec![0];
        let mut max_seqlen: Option<u32> = None;
        let mut min_seqlen: Option<u32> = None;
        for seqlen in seqlens.into_iter() {
            min_seqlen = Some(match min_seqlen {
                None => seqlen,
                Some(min_seqlen) => min_seqlen.min(seqlen),
            });
            max_seqlen = Some(match max_seqlen {
                None => seqlen,
                Some(max_seqlen) => max_seqlen.max(seqlen),
            });
            seqstart_py.push(seqstart_py[seqstart_py.len() - 1] + seqlen);
        }
        let arr: &[f64] = &seqstart_py.iter().map(|x| (*x).into()).collect::<Vec<_>>()[..];
        let seqstart = Tensor::new(arr, &device).map_err(APIError::from)?;
        Ok(Self::new(
            seqstart,
            max_seqlen.unwrap(),
            min_seqlen.unwrap(),
            seqstart_py,
        ))
    }

    fn intervals(&self) -> Box<dyn Iterator<Item = (u32, &u32)>> {
        Box::new(zip(self.seqstart_py, &self.seqstart_py[1..]))
    }
}

pub struct BlockDiagonalCausalMask {
    q_seqinfo: SeqLenInfo,
    k_seqinfo: SeqLenInfo,
    _batch_sizes: Option<Vec<usize>>,
}

impl BlockDiagonalCausalMask {
    fn new(q_seqinfo: SeqLenInfo, k_seqinfo: SeqLenInfo, _batch_sizes: Option<Vec<usize>>) -> Self {
        Self {
            q_seqinfo,
            k_seqinfo,
            _batch_sizes,
        }
    }
}

impl AttentionBias for BlockDiagonalCausalMask {
    /// Queries and Keys are each divided into the same number of blocks.
    /// A query Q in block i cannot attend to a key which is not in block i,
    /// nor one which is farther from the initial key in block i than Q
    /// is from the initial query in block i.
    fn materialize(&self, shape: &Shape, dtype: DType, device: Device) -> Result<Tensor, APIError> {
        //use Tensor::empty, huggingface/candle#1374
        let mask = Tensor::new(
            &shape.dims().iter().map(|x| (*x) as u32).collect::<Vec<_>>()[2..],
            &device,
        )
        .map_err(APIError::from)?
        .to_dtype(dtype)
        .map_err(APIError::from)?;

        for (i, ((q_start, q_end), (k_start, k_end))) in zip(self.q_seqinfo.intervals(), self.k_seqinfo.intervals()).enumerate() {

        }
        todo!()
    }

    fn from_seqlens(
        &self,
        q_seqlen: Vec<u32>,
        kv_seqlen: Option<Vec<u32>>,
        dtype: DType,
        device: Device,
    ) -> Result<Box<dyn AttentionBias>, APIError> {
        assert!(kv_seqlen.is_none() || q_seqlen.len() == kv_seqlen.unwrap().len());
        let q_seqinfo = SeqLenInfo::from_seqlens(q_seqlen.into_iter(), dtype, device)
            .map_err(APIError::from)?;
        let k_seqinfo = if kv_seqlen.is_none() || q_seqlen == kv_seqlen.unwrap() {
            q_seqinfo
        } else {
            SeqLenInfo::from_seqlens(kv_seqlen.unwrap().into_iter(), dtype, device)
                .map_err(APIError::from)?
        };
        Ok(Box::new(Self::new(q_seqinfo, k_seqinfo, None)))
    }
}
