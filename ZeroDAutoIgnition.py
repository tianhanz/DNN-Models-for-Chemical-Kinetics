from modelclass import ModelClass
import os
import json


def main(modelname, epoch):
    modeldir = os.path.join('Model', 'model'+modelname, 'checkpoint')
    settingPath = os.path.join(modeldir, 'settings{}.json'.format(epoch))
    modelPath = os.path.join(modeldir, 'model{}.pt'.format(epoch))
    with open(settingPath, 'r') as f:
        args = json.load(f)

    model = ModelClass()
    model.buildModel(args)
    model.loadModel(modelPath, device='cpu')
    mech_path = os.path.join('Chem', 'ESH2.yaml')
    fuel, reactor = 'H2', 'constP'  # constant pressure reactor
    Builtin_t = 1e-8  # time step size for Cantera

    for T in [1100, 1200, 1300, 1400, 1500, 1600]:  # initial temperature array
        P = 1  # initial P
        Phi = 1.0  # initial phi
        print('T: %sK P: %satm Phi: %s' % (T, P, Phi))
        model.simulateEvolution(
            Phi=Phi,
            T=T,
            P=P,
            n_step=500,  # 500 steps mean 0.5ms since DNN time step size delta_t=1e-6
            plotAll=True,  # if True: plot full evolutions, including T, P, Y;  else: plot T only
            epoch=epoch,
            modelname=modelname,
            mech_path=mech_path,
            fuel=fuel,
            reactor=reactor,
            builtin_t=Builtin_t)
    print('Zero-dimensional auto-iginition results will be shown in ./Model/model%s/pic/pictmp' % modelname)


if __name__ == "__main__":
    modellist = ['528w_msDsH2_constP', '642w_msDsClipESH2_constP']
    modelname = modellist[1]
    epoch = 5000
    main(modelname, epoch)
