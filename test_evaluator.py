from molgenbench.pipeline.evaluator import Evaluator


if __name__ == "__main__":
    evaluator = Evaluator(
        [
            # "Validity", 
            # "QED",
            # "SA",
            # "Uniqueness",
            # "Diversity",
            # "PoseBuster",
            # "StrainEnergy",
            "RMSD"
         ]
    )
    evaluator.run(
        root_dir="./TestSamples",
        model_name="PocketFlow_generated_molecules",
        round="Round1",
        mode="De_novo_Results"
    )
    # evaluator.run(
    #     root_dir="./TestSamples",
    #     model_name="DeleteHit2Lead(CrossDock)_Hit_to_Lead",
    #     round="Round1",
    #     mode="Hit_to_Lead_Results"
    # )


