use std::{
    fs::File,
    io::{self, BufReader, Read, Seek, SeekFrom},
    path::Path,
};

const MAX_METADATA_ENTRIES: u64 = 1_000_000;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARRAY_ITEMS: u64 = 10_000_000;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelMetadata {
    pub architecture: Option<String>,
    pub block_count: Option<u32>,
    pub context_length: Option<u32>,
    pub embedding_length: Option<u32>,
    pub attention_head_count: Option<u32>,
    pub attention_head_count_kv: Option<u32>,
}

pub fn read_model_metadata(path: &Path) -> io::Result<ModelMetadata> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != *b"GGUF" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a GGUF file",
        ));
    }
    let version = read_u32(&mut reader)?;
    if !(2..=3).contains(&version) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported GGUF version {version}"),
        ));
    }
    let _tensor_count = read_u64(&mut reader)?;
    let metadata_count = read_u64(&mut reader)?;
    if metadata_count > MAX_METADATA_ENTRIES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "GGUF metadata entry limit exceeded",
        ));
    }

    let mut metadata = ModelMetadata::default();
    for _ in 0..metadata_count {
        let key = read_string(&mut reader)?;
        let value_type = read_u32(&mut reader)?;
        let wanted = key == "general.architecture"
            || key.ends_with(".block_count")
            || key.ends_with(".context_length")
            || key.ends_with(".embedding_length")
            || key.ends_with(".attention.head_count")
            || key.ends_with(".attention.head_count_kv");
        if !wanted {
            skip_value(&mut reader, value_type)?;
            continue;
        }
        if key == "general.architecture" {
            metadata.architecture = read_string_value(&mut reader, value_type)?;
            continue;
        }
        let value = read_integer_value(&mut reader, value_type)?;
        if key.ends_with(".block_count") {
            metadata.block_count = value;
        } else if key.ends_with(".context_length") {
            metadata.context_length = value;
        } else if key.ends_with(".embedding_length") {
            metadata.embedding_length = value;
        } else if key.ends_with(".attention.head_count_kv") {
            metadata.attention_head_count_kv = value;
        } else if key.ends_with(".attention.head_count") {
            metadata.attention_head_count = value;
        }
    }
    Ok(metadata)
}

fn read_string_value<R: Read + Seek>(
    reader: &mut R,
    value_type: u32,
) -> io::Result<Option<String>> {
    if value_type == 8 {
        read_string(reader).map(Some)
    } else {
        skip_value(reader, value_type)?;
        Ok(None)
    }
}

fn read_integer_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> io::Result<Option<u32>> {
    let value = match value_type {
        0 => read_u8(reader)? as u64,
        2 => read_u16(reader)? as u64,
        4 => read_u32(reader)? as u64,
        10 => read_u64(reader)?,
        1 => read_i8(reader)?.max(0) as u64,
        3 => read_i16(reader)?.max(0) as u64,
        5 => read_i32(reader)?.max(0) as u64,
        11 => read_i64(reader)?.max(0) as u64,
        _ => {
            skip_value(reader, value_type)?;
            return Ok(None);
        }
    };
    Ok(u32::try_from(value).ok())
}

fn skip_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> io::Result<()> {
    match value_type {
        0 | 1 | 7 => seek_forward(reader, 1),
        2 | 3 => seek_forward(reader, 2),
        4..=6 => seek_forward(reader, 4),
        10..=12 => seek_forward(reader, 8),
        8 => {
            let length = read_u64(reader)?;
            checked_length(length, MAX_STRING_BYTES, "GGUF string")?;
            seek_forward(reader, length)
        }
        9 => {
            let element_type = read_u32(reader)?;
            let count = read_u64(reader)?;
            checked_length(count, MAX_ARRAY_ITEMS, "GGUF array")?;
            if let Some(size) = fixed_type_size(element_type) {
                seek_forward(
                    reader,
                    count.checked_mul(size).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "GGUF array size overflow")
                    })?,
                )
            } else if element_type == 8 {
                for _ in 0..count {
                    let length = read_u64(reader)?;
                    checked_length(length, MAX_STRING_BYTES, "GGUF string")?;
                    seek_forward(reader, length)?;
                }
                Ok(())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unsupported nested GGUF metadata array",
                ))
            }
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown GGUF metadata type {value_type}"),
        )),
    }
}

fn fixed_type_size(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn checked_length(value: u64, maximum: u64, label: &str) -> io::Result<()> {
    if value > maximum {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} limit exceeded"),
        ))
    } else {
        Ok(())
    }
}

fn read_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let length = read_u64(reader)?;
    checked_length(length, MAX_STRING_BYTES, "GGUF string")?;
    let mut bytes = vec![0; length as usize];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))
}

fn seek_forward<R: Seek>(reader: &mut R, amount: u64) -> io::Result<()> {
    let amount = i64::try_from(amount)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "GGUF offset is too large"))?;
    reader.seek(SeekFrom::Current(amount)).map(|_| ())
}

macro_rules! read_number {
    ($name:ident, $type:ty, $size:expr) => {
        fn $name<R: Read>(reader: &mut R) -> io::Result<$type> {
            let mut bytes = [0u8; $size];
            reader.read_exact(&mut bytes)?;
            Ok(<$type>::from_le_bytes(bytes))
        }
    };
}

read_number!(read_u8, u8, 1);
read_number!(read_i8, i8, 1);
read_number!(read_u16, u16, 2);
read_number!(read_i16, i16, 2);
read_number!(read_u32, u32, 4);
read_number!(read_i32, i32, 4);
read_number!(read_u64, u64, 8);
read_number!(read_i64, i64, 8);
