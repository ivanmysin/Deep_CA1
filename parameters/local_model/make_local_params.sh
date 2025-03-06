python ca1_pyrs.py
python ca1_interneurons.py
python ca1_gens.py
python join_neurons.py
echo "Params of population is generated"
cd ../
python set_connections.py
echo "Params of connections is generated"
cd ./local_model
python remove_generators_without_postsynapses.py
cd ../
python set_connections.py
echo "Params are ready"