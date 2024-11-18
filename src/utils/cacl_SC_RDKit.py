import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig

# Set up features to use in FeatureMap
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
fdef = AllChem.BuildFeatureFactory(fdefName)

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable',
        'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')


def get_FeatureMapScore(query_mol, ref_mol):
    featLists = []
    for m in [query_mol, ref_mol]:
        rawFeats = fdef.GetFeaturesForMol(m)
        # filter that list down to only include the ones we're intereted in
        featLists.append([f for f in rawFeats if f.GetFamily() in keep]) # [[query_feat], [ref_feat]]
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists] # fms = [[query_fms], [ref_fms]]
    fms[0].scoreMode = FeatMaps.FeatMapScoreMode.Best # only query_fms
    fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))

    return fm_score


def calc_SC_RDKit_score(query_mol, ref_mol):
    fm_score = get_FeatureMapScore(query_mol, ref_mol)

    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(query_mol, ref_mol,
                                                     allowReordering=False)
    SC_RDKit_score = 0.5 * fm_score + 0.5 * (1 - protrude_dist)

    return SC_RDKit_score


# optimize for multi scrdkit calculation
def cache_ref_feats(ref_mols):
    ref_feats_cache = []
    for ref_mol in ref_mols:
        raw_feats = fdef.GetFeaturesForMol(ref_mol)
        # filter that list down to only include the ones we're intereted in
        ref_feats_cache.append([f for f in raw_feats if f.GetFamily() in keep])
    return ref_feats_cache


def get_FeatureMapScore_new(query_mol, ref_feats):
    featLists = []
    query_raw_feats = fdef.GetFeaturesForMol(query_mol)
    query_feats = [f for f in query_raw_feats if f.GetFamily() in keep]
    featLists.append(query_feats)
    
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists]
    fms[0].scoreMode = FeatMaps.FeatMapScoreMode.Best
    fm_score = fms[0].ScoreFeats(ref_feats) / min(fms[0].GetNumFeatures(), len(ref_feats))

    return fm_score


def calc_SC_RDKit_score_new(query_mol, ref_mol, ref_feats):
    fm_score = get_FeatureMapScore_new(query_mol, ref_feats)

    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(query_mol, ref_mol,
                                                     allowReordering=False)
    SC_RDKit_score = 0.5 * fm_score + 0.5 * (1 - protrude_dist)

    return SC_RDKit_score