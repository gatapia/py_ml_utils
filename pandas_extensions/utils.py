import gzip, scipy
import pandas as pd, numpy as np

def chunked_iterator(df, chunk_size=1000000):
  start = 0
  while True:
    subset = df[start:start+chunk_size]
    start += chunk_size
    for r in subset.iterrows():
      yield r[1]
    if len(subset) < chunk_size: break

def get_write_file_stream(file):
  return gzip.GzipFile(file, 'wb') if file.endswith('.gz') else open(file, "wb")

def get_optimal_numeric_type(dtype, min, max, aggresiveness=0):
  dtype = str(dtype)
  is_int = dtype.startswith('int')
  if min >= 0 and is_int:
    '''
    uint8 Unsigned integer (0 to 255)
    uint16  Unsigned integer (0 to 65535)
    uint32  Unsigned integer (0 to 4294967295)
    uint64  Unsigned integer (0 to 18446744073709551615)
    '''
    if max <= 255: return 'uint8'
    if max <= 65535: return 'uint16'
    if max <= 4294967295: return 'uint32'
    if max <= 18446744073709551615: return 'uint64'
    raise Exception(`max` + ' is too large')
  elif is_int:
    '''
    int8 Byte (-128 to 127)
    int16 Integer (-32768 to 32767)
    int32 Integer (-2147483648 to 2147483647)
    int64 Integer (-9223372036854775808 to 9223372036854775807)
    '''
    if min >= -128 and max <= 127: return 'int8'
    if min >= -32768 and max <= 32767: return 'int16'
    if min >= -2147483648 and max <= 2147483647: return 'int32'
    if min >= -9223372036854775808 and max <= 9223372036854775807: return 'int64'
    raise Exception(`min` + ' and ' + `max` + ' are out of supported range')
  else:
    '''
    float16 Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
    float32 Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    float64 Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    '''
    if not dtype.startswith('float'): raise Exception('Unsupported type: ' + dtype)
    current = int(dtype[-2:])
    if aggresiveness == 0: return dtype
    if aggresiveness == 1: 
      if current == 64: return 'float32'
      elif current <= 32: return 'float16'
      elif current == 16: return 'float16'
      else: raise Exception('Unsupported type: ' + dtype)
    if aggresiveness >= 2: return 'float16'  

def get_col_aggregate(col, mode):
  '''
  col: A pandas column
  mode: One of <constant>|mode|mean|iqm|median|min|max
  '''
  if type(mode) != str: return mode
  if mode == 'mode': return col.mode().iget(0) 
  if mode == 'mean': return col.mean()
  if mode == 'iqm': 
    to_replace = np.isnan(col) | np.isinf(col) | np.isneginf(col)
    if np.any(to_replace): 
      col[to_replace] = col.mean()
      iqm = np.mean(np.percentile(col, [75 ,25]))
      col[to_replace] = np.nan
      return iqm
    return np.mean(np.percentile(col, [75 ,25]))
  if mode == 'median': return col.median()
  if mode == 'min': return col.min()
  if mode == 'max': return col.max()
  if mode == 'max+1': return col.max()+1
  return mode

def is_sparse(o):
  return type(o) is pd.sparse.frame.SparseDataFrame or \
    type(o) is pd.sparse.series.SparseSeries or \
    scipy.sparse.issparse(o)

def create_df_from_templage(template, data, index=None):
  df = pd.DataFrame(columns=template.columns, data=data, index=index)
  for c in template.columns:
    if template[c].dtype != df[c].dtype: 
      df[c] = df[c].astype(template[c].dtype)
  return df

