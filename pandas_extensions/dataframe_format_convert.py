import utils

def __df_to_lines(df, 
    out_file_or_y=None, 
    y=None, 
    weights=None, 
    convert_zero_ys=True,
    output_categorical_value=True,
    tag_feature_sets=True,
    col_index_start=0,
    sort_feature_indexes=False):    
  columns_indexes = {}
  max_col = {'index':0}
  out_file = out_file_or_y if type(out_file_or_y) is str else None
  
  if y is None and out_file_or_y is not None and out_file is None: 
    y = out_file_or_y
  if y is not None and hasattr(y, 'values'): y = y.values

  def get_col_index(name):
    if name not in columns_indexes:
      columns_indexes[name] = max_col['index']
      max_col['index'] += 1
    return str(col_index_start + columns_indexes[name])

  def impl(outfile):
    def add_cols(new_line, columns, is_numerical):
      if len(columns) == 0: return
      if tag_feature_sets: new_line.append('|' + ('n' if is_numerical else 'c'))
      for c in columns:        
        val = row[c] 
        if val == 0: continue        
        if not is_numerical:
          name = c + '_' + str(val)
          if output_categorical_value: line = get_col_index(name) + ':1'
          else: 
            line = get_col_index(name)
        else:           
          line = get_col_index(c) + ':' + str(val)
        new_line.append(line)
      
    lines = []  
    for idx, row in enumerate(utils.chunked_iterator(df)):
      label = '1.0' if y is None or idx >= len(y) else str(float(y[idx]))
      if convert_zero_ys and label == '0.0': label = '-1.0'
      if weights is not None and idx < len(weights):      
        w = weights[idx]
        if w != 1: label += ' ' + `w`
        label += ' \'' + `idx`
      
      new_line = []
      add_cols(new_line, df.numericals(), True)
      add_cols(new_line, df.categoricals() + df.indexes() + df.binaries(), False)
      if sort_feature_indexes:
        new_line = sorted(new_line, key=lambda v: int(v.split(':')[0]))
      line = ' '.join([label] + new_line)
  
      if outfile: outfile.write(line + '\n')
      else: lines.append(line)
    return lines
  
  if out_file:
    with utils.get_write_file_stream(out_file) as outfile:    
      return impl(outfile)
  else: 
    return impl(None)

def _df_to_vw(self, out_file_or_y=None, y=None, weights=None):    
  return __df_to_lines(self, out_file_or_y, y, weights, 
      convert_zero_ys=True,
      output_categorical_value=False,
      tag_feature_sets=True)  

def _df_to_svmlight(self, out_file_or_y=None, y=None):
  return __df_to_lines(self, out_file_or_y, y, None,
      convert_zero_ys=True,
      output_categorical_value=True,
      tag_feature_sets=False,
      col_index_start=1,
      sort_feature_indexes=True)

def _df_to_libfm(self, out_file_or_y=None, y=None):
  return __df_to_lines(self, out_file_or_y, y, None,
      convert_zero_ys=False,
      output_categorical_value=True,
      tag_feature_sets=False)

def _df_to_libffm(self, out_file_or_y=None, y=None):
  out_file = out_file_or_y if type(out_file_or_y) is str else None  
  if y is None and out_file_or_y is not None and out_file is None: 
    y = out_file_or_y
  if hasattr(y, 'values'): y = y.values
  
  lines = []    
  outfile = None
  if out_file is not None: outfile = utils.get_write_file_stream(out_file)
  categoricals = self.categoricals() + self.indexes() + self.binaries()
  if len(categoricals) > 0: raise Exception('categoricals not currently supported')
  numericals = self.numericals()
  for idx, row in enumerate(utils.chunked_iterator(self)):
    label = '0' if y is None or idx >= len(y) else str(int(y[idx]))
    new_line = [label]     
    for col_idx, c in enumerate(numericals):
      cis = str(col_idx)
      new_line.append(cis + ':0:' + `0 if row[c] == 0  else row[c]`)
    line = ' '.join(new_line)
    if outfile is not None: outfile.write(line + '\n')
    else: lines.append(line)
  if outfile:
    outfile.close()
    return None
  else: return lines
