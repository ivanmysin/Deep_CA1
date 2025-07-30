import pandas as pd

output_file_path = '/home/ivan/PycharmProjects/Deep_CA1/parameters/PetentialConnections.csv'
filepath = '/home/ivan/Data/hippocampome/Connections.csv'

conns_df = pd.read_csv(filepath, header=0, sep=',')
post_synaptic = conns_df.columns.to_list()
post_synaptic.remove('Presynaptic')

# print(conns_df['Presynaptic'])
# print(post_synaptic)

output_str = '''Presynaptic Neuron Type,Postsynaptic Neuron Type\n'''

for pre_type in conns_df['Presynaptic']:
    # print(pre_type)

    for post_type in post_synaptic:

        try:
            conn_val = conns_df[conns_df['Presynaptic'] == pre_type][post_type].values[0]
        except IndexError:
            continue

        post_type = post_type.split(':')[1]
        post_type = post_type.replace('*', '')
        pre_type = pre_type.replace('*', '')

        if conn_val in ['pxc', 'phc']:
            output_str += f'{pre_type},{post_type}\n'
            #print(pre_type, post_type)

with open(output_file_path, 'w') as output_file:
    output_file.write(output_str)



