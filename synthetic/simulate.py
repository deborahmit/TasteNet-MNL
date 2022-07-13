import torch
import pickle
from collections import OrderedDict
from evaluation import value_of_x, RMSE, ABSE, RE


def error_of_vot(best_model, dic_z, input_z, params):
    ## result is converted to dollar per hour!
    pred_vots = []

    with torch.no_grad():
        for key in input_z:
            pred_vots.append((-best_model.taste(input_z[key])[:, 0] * 60).flatten())

    true_vots = []
    for key in dic_z:
        true_vots.append((-value_of_x(params['A0_time'], params['A1_time'], params['A2_time'], dic_z[key]) * 60).flatten())

    rmse = RMSE(pred_vots, true_vots).item()
    mabse = ABSE(pred_vots, true_vots).item()
    re = RE(pred_vots, true_vots).item()

    return pred_vots, true_vots, rmse, mabse, re


def error_of_vowt(best_model, dic_z, input_z, params):
    ## result is converted to dollar per hour!
    pred_vowts = []

    with torch.no_grad():
        for key in input_z:
            pred_vowts.append((-best_model.taste(input_z[key])[:, 1] * 60).flatten())

    true_vowts = []
    for key in dic_z:
        true_vowts.append(
            (-value_of_x(params['A0_wait'], params['A1_wait'], params['A2_wait'], dic_z[key]) * 60).flatten()
        )

    rmse = RMSE(pred_vowts, true_vowts).item()
    mabse = ABSE(pred_vowts, true_vowts).item()
    re = RE(pred_vowts, true_vowts).item()

    return pred_vowts, true_vowts, rmse, mabse, re


# True parameters
params = pickle.load(open("../toy_data/params.pkl", "rb"))

# Generate z
inc = torch.arange(0.0, 1.0, 0.02).reshape(-1, 1)
n = len(inc)
fulltime = torch.ones(n, 1)
nofulltime = torch.zeros(n, 1)
flex = torch.ones(n, 1)
noflex = torch.zeros(n, 1)


dic_z = OrderedDict([("full_flex", torch.cat((inc, fulltime, flex), dim=1)),
                     ("full_noflex", torch.cat((inc, fulltime, noflex), dim=1)),
                     ("nofull_flex", torch.cat((inc, nofulltime, flex), dim=1)),
                     ("nofull_noflex", torch.cat((inc, nofulltime, noflex), dim=1))])

dic_z_z01 = OrderedDict([("full_flex", torch.cat((inc, fulltime, flex, inc * fulltime), dim=1)),
                         ("full_noflex", torch.cat((inc, fulltime, noflex, inc * fulltime), dim=1)),
                         ("nofull_flex", torch.cat((inc, nofulltime, flex, inc * nofulltime), dim=1)),
                         ("nofull_noflex", torch.cat((inc, nofulltime, noflex, inc * nofulltime), dim=1))])

dic_z_zall = OrderedDict(
    [("full_flex", torch.cat((inc, fulltime, flex, inc * fulltime, inc * flex, fulltime * flex), dim=1)),
     ("full_noflex", torch.cat((inc, fulltime, noflex, inc * fulltime, inc * noflex, fulltime * noflex), dim=1)),
     ("nofull_flex", torch.cat((inc, nofulltime, flex, inc * nofulltime, inc * flex, nofulltime * flex), dim=1)),
     ("nofull_noflex",
      torch.cat((inc, nofulltime, noflex, inc * nofulltime, inc * noflex, noflex * nofulltime), dim=1))])