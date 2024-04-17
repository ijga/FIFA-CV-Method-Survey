import numpy as np
even = 1.5707964897155762


def calcHomog(theta):

    even = 1.5707964897155762
    goalkick_offset = 0.0436332
    value = theta - even
    print(value)
    
    x1 = 73.655650176915*(value)**2 + 0.833519727
    x2 = 355.84689145836*(value)**2 + 1.93684006
    tx = -24322.6*(value) + 82.2433
    y1 = -5.79838*(value) + -0.00138694
    y2 = 338.61744892211*(value)**2 + 4.30087726
    ty = -287845.3827689*(value)**2 - 1228.02498
    p1 = -0.000318411*(value) + 0.00000383338
    p2 = 0.16781575188979*(value)**2 + 0.00201857912

    # [quadratic, quadratic, linear
    # linear, quadratic, quadratic
    # linear, quadratic, constant]

    homog = np.array([[ x1, x2, tx],
                        [y1, y2, ty],
                        [p1,  p2,  1.00000000e+00]])

    return homog


print(calcHomog(-0.0436332 + even))
print(calcHomog(even))
print(calcHomog(0.0436332 + even))


Hr = np.array([[ 9.73749489e-01,  2.61432135e+00,  1.10399484e+03],
                [ 2.50914243e-01,  4.94555617e+00, -1.77604118e+03],
                [ 1.96328735e-05,  2.33807617e-03,  1.00000000e+00]]) # -2.5 degrees

Hc = np.array([[ 8.33519727e-01,  1.93684006e+00,  1.61288507e+02],
                [ 1.43620104e-05,  4.30087726e+00, -1.22802498e+03],
                [ 2.09970635e-08,  2.01857912e-03,  1.00000000e+00]])

Hl = np.array([[ 9.60447704e-01,  1.95174080e+00, -1.01855334e+03],
                [-2.55089434e-01,  5.10054596e+00, -1.37582741e+03],
                [-8.15372620e-06,  2.48185855e-03,  1.00000000e+00]]) # +2.5 degrees