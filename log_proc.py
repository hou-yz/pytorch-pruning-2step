import json
from openpyxl import Workbook


with open('./checkpoint/log_original.json') as fp:
    original_data=json.load(fp)
with open('./checkpoint/log_pruned.json') as fp:
    pruned_data=json.load(fp)

layer_cfg=['']
acc=['acc_original']
delta_t=['time_original']
delta_t_computations=['']
bandwidth=['bandwidth_original']
all_conv_computations=['']
for data in original_data:
    layer_cfg.append(data['conv_index'])
    acc.append(data['acc'])
    delta_t.append(data['delta_t'])
    delta_t_computations.append(data['delta_t_computations'])
    bandwidth.append(data['bandwidth'])
    all_conv_computations.append(data['all_conv_computations'])

wb = Workbook()
ws = wb.create_sheet("original", 0)
ws.append(layer_cfg)
ws.append(acc)
ws.append(delta_t)
ws.append(delta_t_computations)
ws.append(bandwidth)
ws.append(all_conv_computations)

layer_cfg=['']
acc=['acc_pruned']
delta_t=['time_pruned']
delta_t_computations=['']
bandwidth=['bandwidth_pruned']
all_conv_computations=['']
for data in pruned_data:
    layer_cfg.append(data['conv_index'])
    acc.append(data['acc'])
    delta_t.append(data['delta_t'])
    delta_t_computations.append(data['delta_t_computations'])
    bandwidth.append(data['bandwidth'])
    all_conv_computations.append(data['all_conv_computations'])

ws = wb.create_sheet("pruned", 0)
ws.append(layer_cfg)
ws.append(acc)
ws.append(delta_t)
ws.append(delta_t_computations)
ws.append(bandwidth)
ws.append(all_conv_computations)

wb.save('log.xlsx')

