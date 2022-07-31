# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:34:47 2020
@author: Tianhan Zhang
@email: 
"""
import cantera as ct
from torch.nn.modules import Module
from torch import nn
import torch
import matplotlib.pyplot as plt
import os
import math
import json
import re
import numpy as np
from copy import deepcopy
# plot
import matplotlib

matplotlib.use('AGG')
# pytorch
# cantera


# customize activation funciton GELU
class MyGELU(Module):
    def __init__(self):
        super(MyGELU, self).__init__()

    def forward(self, x):
        torch_PI = 3.1415926536
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / torch_PI) * (x + 0.044715 * torch.pow(x, 3))))


class Network(nn.Module):
    def __init__(self, args, Actfuns):
        super(Network, self).__init__()
        neurons = args['layers']
        self.depth = len(neurons) - 1
        self.actfun = Actfuns[args['actfun']]
        self.layers = []
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(self.actfun)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))  # last layer
        self.fc = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.fc(x)
        return x


class ModelClass():
    def __init__(self, work_dir=None, **kwargs):
        self.work_dir = os.getcwd() if work_dir is None else work_dir

    def buildModel(self, args):  # create DNN
        # activation funcitons
        Actfuns = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'mygelu': MyGELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'CEL': nn.CrossEntropyLoss(),
        }
        self.net = Network(args, Actfuns)
        print("the neural network is created. Network type: %s " %
              args['net_type'])

    def loadModel(self, modelDir, device='cuda:0'):
        self.net.load_state_dict(torch.load(modelDir, map_location=device))
        print("%s is loaded on %s" % (modelDir, device))

    def ctOneStep(self, state, delta_t, mech_path, reactor, builtin_t):
        '''
        cantera one-step (For instance, H2/air mechanism)
        state: 1x(n+2) numpy array input state, including T,P(atm),Y
        delta_t: state after delta_t
        return :1x(n+2) numpy array output state
        '''
        gas = ct.Solution(mech_path)
        n = gas.n_species  # gas species
        state = state.reshape(-1, n + 2)
        gas.TPY = state[0, 0], state[0, 1] * ct.one_atm, state[0, 2:]
        if reactor == 'constV':  # constant Volume reactor
            r = ct.IdealGasReactor(gas)
        if reactor == 'constP':  # constant Pressure reactor
            r = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([r])
        sim.max_time_step = builtin_t
        sim.advance(delta_t)
        Y = gas.Y  # mass fraction
        state_out = np.c_[gas.T, gas.P / ct.one_atm,
                          Y.reshape(-1, n)]  # (1,n+2)
        return state_out

    def netOneStep(self, state, args):
        '''
        state: np.array; supposed to be [T, P(atm), Y_i]
        return: np.array; state after delta_t
        '''
        lamda = args['power_transform']
        delta_t = args['delta_t']
        state_bct = state.reshape(-1, args['dim'])
        state_bct[:, 2:] = (state_bct[:, 2:]**(lamda) - 1) / lamda  # TPY
        Xmu = np.array(args['Xmu'])
        Xstd = np.array(args['Xstd'])
        state_normalized = (state_bct - Xmu) / Xstd
        state_normalized = torch.from_numpy(state_normalized).float()
        output_normalized = self.net(state_normalized)
        output_normalized = output_normalized.detach().cpu().numpy()
        output = output_normalized * args['Ystd'] + args['Ymu']
        output_bct = output * delta_t + state_bct
        output_bct[:, 2:] = (lamda * output_bct[:, 2:] + 1)**(1 / lamda)  #
        return output_bct

    def norm(self, X):
        # normalization by summation
        X = X / np.sum(X, axis=1, keepdims=True)
        return X

    def OneStep(self, args, X, Y, mech_path, reactor, builtin_t):
        '''
        X: input of DNN
        Y: initial state for cantera 
        '''
        state0 = Y.copy()
        # cantera prediction
        pred = self.netOneStep(X, args)
        # cantera output
        state_out = self.ctOneStep(state0, args['delta_t'], mech_path, reactor,
                                   builtin_t)
        return pred, state_out

    def plotTemperature(self, args, Phi, T, P, real, net, epoch, pic_path,
                        modelname):
        '''
        plot ignition curve simulated by cantera and DNN, then save pictures
        real: true state vector calculated by Cantera
        net: prediction state of neural networks
        '''
        n_step = range(0, len(net[:, 0]))
        time = np.array(n_step) * args['delta_t'] * 1e3
        p1, = plt.plot(time,
                       real[:, 0],
                       color="chocolate",
                       linewidth=1.5,
                       alpha=1)
        p2, = plt.plot(time,
                       net[:, 0],
                       color="green",
                       linewidth=1.5,
                       linestyle='-.',
                       alpha=1)
        plt.xlabel("time(ms)")
        plt.ylabel('Temperature (K)')
        plt.legend([p1, p2], ["Cantera", 'DNN'], loc="lower right")
        pic_name = os.path.join(
            pic_path,
            '{}_Phi={}_T={}_P={}_epoch={}.png'.format(modelname, Phi, T, P,
                                                      epoch))
        plt.savefig(pic_name, dpi=500)
        plt.close()

    def latex(self, str):
        # 'H2O'-->'$H_2O$'
        content = re.findall('\d', str)
        numList = list(set(content))
        for number in numList:
            str = re.sub(number, '_' + number, str)
        return '$' + str + '$'

    def component(self, num):
        # 'T,P,Y'
        TP = ['T', 'P']
        gas_TPY = TP + self.gas.species_names
        str = gas_TPY[num]
        return self.latex(str)

    def plotAll(self, args, Phi, T, P, real, net, epoch, pic_path, modelname):
        n_step = range(0, len(net[:, 0]))
        time = np.array(n_step) * args['delta_t'] * 1e3
        fig = plt.figure(figsize=(12.8, 9.6))
        for i in range(2 + self.gas.n_species):
            num = i % 9 + 1
            ax = fig.add_subplot(3, 3, num)
            if i == 0 or i == 1:
                p1, = plt.plot(time,
                               real[:, i],
                               color="chocolate",
                               linewidth=2,
                               alpha=1)
                p2, = plt.plot(time,
                               net[:, i],
                               color="green",
                               linewidth=2,
                               linestyle='-.',
                               alpha=1)

            else:
                p1, = plt.semilogy(time,
                                   real[:, i],
                                   color="chocolate",
                                   linewidth=2,
                                   alpha=1)
                p2, = plt.semilogy(time,
                                   net[:, i],
                                   color="green",
                                   linewidth=2,
                                   linestyle='-.',
                                   alpha=1)
            # ---------------------------------
            if i != 0 and i != 1:
                order = math.floor(np.log10(real[-1, i] + 1e-30))
                low = order - 3
                up = min(0, order + 3)
                plt.ylim(10**low, 10**up)
                numList = [10**(i) for i in range(low, up + 1)]
                plt.yticks(numList)
                plt.ylabel('mass fraction')
            elif i == 0:
                plt.ylabel('Temperature (K)')
            else:
                plt.ylabel('Pressure (atm)')

            # ---------------------------------
            plt.legend([p1, p2], [
                "Cantera",
                'DNN',
            ],
                fontsize=8,
                loc='lower right')
            plt.title(self.component(i))
            plt.xlabel("time(ms)")

            if num == 9:
                plt.tight_layout()
                pic_name = os.path.join(
                    pic_path, '{}_Phi={}_T={}_P={}_epoch={}_all{}.png'.format(
                        modelname, Phi, T, P, epoch, math.ceil(i / 9)))
                plt.savefig(pic_name, dpi=200)
                plt.close()
                fig = plt.figure(figsize=(12.8, 9.6))
            elif i == 1 + self.gas.n_species:
                plt.tight_layout()
                pic_name = os.path.join(
                    pic_path, '{}_Phi={}_T={}_P={}_epoch={}_all{}.png'.format(
                        modelname, Phi, T, P, epoch, math.ceil((i + 1) / 9)))
                plt.savefig(pic_name, dpi=200)
                plt.close()

    def simulateData(self, args, Phi, T, P, n_step, mech_path, fuel, reactor,
                     builtin_t):

        self.gas = ct.Solution(mech_path)
        self.gas.TP = T, P  # warning P(atm)
        self.gas.set_equivalence_ratio(Phi, fuel, 'O2:1.0,N2:3.76')
        prediction = np.c_[self.gas.T, self.gas.P,
                           self.gas.Y.reshape(-1, self.gas.n_species)]
        prediction = prediction.reshape(-1, self.gas.n_species + 2)

        state = deepcopy(prediction)
        for i in range(n_step):
            netinput = prediction[i, :].copy()
            ctinput = state[i, :].copy()
            net_out, state_out = self.OneStep(args, netinput, ctinput,
                                              mech_path, reactor, builtin_t)
            prediction = np.concatenate([prediction, net_out], axis=0)
            state = np.concatenate([state, state_out], axis=0)
        return state, prediction

    def simulateEvolution(
        self,
        Phi,
        T,
        P,
        n_step,
        plotAll,
        epoch,
        modelname,
        mech_path,
        fuel,
        reactor,
        builtin_t,
    ):
        '''
        corresponding function: ctOneStep(), netOneStep(), plotTemperature()
                                plotAll(), simulateData()
        Phi: float; initial equivalence ratio
        T: float; initial Temperature (K)
        P: float; initial Pressure (atm)
        n_step: int; evolution step
        plotAll: bool; if plot all teh dimensions of state 
        mech_path: str; chemical mechnism dir for Cantera
        reactor: str; constP or constV, the reactor type 
        builtin_t: float; Cantera max_time_step (seconds)
        '''
        pic_path = os.path.join('Model', 'model' + modelname, 'pic', 'pictmp')
        if not os.path.exists(pic_path):
            os.mkdir(pic_path)

        # load DNN model
        model_folder = os.path.join('Model', 'model' + modelname, 'checkpoint')
        settings_dir = os.path.join(model_folder,
                                    'settings{}.json'.format(epoch))
        model_dir = os.path.join(model_folder, 'model{}.pt'.format(epoch))

        with open(settings_dir, 'r') as f:
            Args = json.load(f)

        self.net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        ######

        real, net_pred = self.simulateData(Args, Phi, T, P, n_step, mech_path,
                                           fuel, reactor, builtin_t)
        if plotAll == True:
            self.plotAll(Args, Phi, T, P, real, net_pred, epoch, pic_path,
                         modelname)
        if plotAll == False:
            self.plotTemperature(Args, Phi, T, P, real, net_pred, epoch,
                                 pic_path, modelname)
        print('\repoch {:^4} simulation pic generated.'.format(epoch),
              end='\n')
