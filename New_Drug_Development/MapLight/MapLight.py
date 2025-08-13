# AvalonCountFP, HashedMorganFingerprint, ErGFingerprint, rdkit, PretrainedDGLTransformer

import numpy as np
import pandas as pd

from sklearn import preprocessing

from mordred import Calculator, descriptors

from rdkit import Chem
from rdkit import RDLogger

from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from molfeat.trans.pretrained import PretrainedDGLTransformer

from tqdm import tqdm


class scaler:
    def __init__(self, log=False):
        self.log = log
        self.offset = None
        self.scaler = None

    def fit(self, y):
        # make the values non-negative
        self.offset = np.min([np.min(y), 0.0])
        y = y.reshape(-1, 1) - self.offset

        # scale the input data
        if self.log:
            y = np.log10(y + 1.0)

        self.scaler = preprocessing.StandardScaler().fit(y)

    def transform(self, y):
        y = y.reshape(-1, 1) - self.offset

        # scale the input data
        if self.log:
            y = np.log10(y + 1.0)

        y_scale = self.scaler.transform(y)

        return y_scale

    def inverse_transform(self, y_scale):
        y = self.scaler.inverse_transform(y_scale)

        if self.log:
            y = 10.0**y - 1.0

        y = y + self.offset

        return y


def avalon_fingerprint(molecule, n_bits = 1024):
    return GetAvalonCountFP(molecule, n_bits)

def morgan_fingerprint(molecule, n_bits = 1024, radius = 2):
    return GetHashedMorganFingerprint(molecule, radius, n_bits)

def erg_fingerprint(molecule):
    return GetErGFingerprint(molecule)

def rdkit_features(molecule):
    descriptors = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 
        'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 
        'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 
        'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 
        'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 
        'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 
        'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 
        'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge', 
        'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge', 
        'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 
        'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 
        'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 
        'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 
        'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 
        'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 
        'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 
        'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 
        'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 
        'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 
        'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 
        'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 
        'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 
        'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 
        'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 
        'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 
        'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 
        'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 
        'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 
        'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 
        'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 
        'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 
        'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 
        'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 
        'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 
        'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 
        'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 
        'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 
        'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 
        'fr_unbrch_alkane', 'fr_urea', 'qed']

    calculator = MolecularDescriptorCalculator(descriptors)
    
    return calculator.CalcDescriptors(molecule)

def gin_supervised_masking(transformer, molecule):
    return transformer(molecule)


if __name__ == "__main__":

    TRAIN_PATH = "./data/train.csv"
    TEST_PATH = "./data/test.csv"
    SUBMISSION_PATH = "./data/sample_submission.csv"

    RDLogger.DisableLog('rdApp.*')

    train_df = pd.read_csv(TRAIN_PATH)
    smiles = train_df["Canonical_Smiles"]

    transformer = PretrainedDGLTransformer(kine = "gin_supervised_masking", dtype = float)
    mordred_calc = Calculator(descriptors, ignore_3D=True)

    mordred_list = []
    avalon_list = []
    morgan_list = []
    # erg_list = []
    # rdkit_list = []
    gin_list = []

    for smile in tqdm(smiles):
        mol = Chem.MolFromSmiles(smile)
        
        avalon_list.append(avalon_fingerprint(mol).ToList())
        morgan_list.append(morgan_fingerprint(mol).ToList())
        # erg_list.append(erg_fingerprint(mol).tolist())
        # rdkit_list.append(rdkit_features(mol))
        gin_list.append(gin_supervised_masking(transformer, mol).tolist())
        mordred_list.append(mordred_calc(mol))

    avalon_list = np.array(avalon_list)
    morgan_list = np.array(morgan_list)
    # erg_list = np.array(erg_list)
    # rdkit_list = np.array(rdkit_list)
    gin_list = np.squeeze(np.array(gin_list), axis = 1)
    mordred_list = np.array(mordred_list)


    print(avalon_list.shape, morgan_list.shape, gin_list.shape, mordred_list.shape)

    combined = np.concatenate((avalon_list, morgan_list, gin_list, mordred_list), axis = 1)

    print(combined.shape)

    np.savetxt("combined_features.csv", combined, delimiter=",")