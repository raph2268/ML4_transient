from lsst.daf.butler import Butler


from injection import create_catalog_for_all_ccd


repo = "/sps/lsst/groups/transients/HSC/fouchez/RC2_repo/butler.yaml"
collection = 'run/ssp_ud_cosmos/step5'
butler = Butler(repo, collections=collection)
registry = butler.registry
datasetRefs = registry.queryDatasets(datasetType='sourceTable',collections=collection)

create_catalog_for_all_ccd(datasetRefs, butler, 'i', True, save_filename='catalog_i')
