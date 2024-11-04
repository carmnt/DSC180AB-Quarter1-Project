'''
etl.py contains functions used to download DataFrames containing assemblies 
and chromosomes for specific genomes.
'''

from datasets import load_dataset

ds = load_dataset("songlab/genomes-brassicales-balanced-v1")

def get_assembly_and_chrom(dataset, assembly, chrom):
    """
    Returns a dataset filtered by 'assembly' and 'chrom' values.
    """
    return dataset.filter(lambda x: x['assembly'] == assembly and x['chrom'] == chrom)

def get_data(assemblies, chroms, outpath):
    '''
    Downloads Dataset and saves them as CSVs at the specified output directory for the given assemblies and chroms.

    :param: assemblies: a list of assemblies to collect
    :param: chroms: a list of chroms to collect
    :param: outpath: the directory in which to save the data.
    '''
    for assembly in assemblies:
        for chrom in chroms:
            data = get_assembly_and_chrom(ds, assemblies, chroms)
            data.to_csv(os.path.join(outpath, f'{assembly}-{chrom}.csv'))