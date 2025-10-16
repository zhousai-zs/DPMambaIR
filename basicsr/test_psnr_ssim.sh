#! /bin/bash
# python test.py -opt /zs/Restormer/basicsr/options/test/test_DPMambaIR.yml
# python test.py -opt /zs/Restormer/basicsr/options/test/test_AdaIR.yml
# python test.py -opt /zs/Restormer/basicsr/options/test/test_OneRestore.yml
# python test.py -opt /zs/Restormer/basicsr/options/test/test_PromptIR.yml
# python test.py -opt /zs/Restormer/basicsr/options/test/test_Restormer.yml

python test.py -opt /zs/Restormer/basicsr/options/test/test_DPMambaIRBi.yml
python test.py -opt /zs/Restormer/basicsr/options/test/test_DPMambaIRBiPromptAB.yml
python test.py -opt /zs/Restormer/basicsr/options/test/test_DPMambaIRnewDE.yml

