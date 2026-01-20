def write_vtk(filename, coords, scalars, precision=2):
    n_points = coords.shape[0]
    with open(filename, 'w') as f:
        f.write('# vtk DataFile Version 2.0\n')
        f.write('VTK from Python\n')
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')
        f.write(f'POINTS {n_points} double\n')
        for i in range(n_points):
            f.write(f'{coords[i,0]:.{precision}f} {coords[i,1]:.{precision}f} {coords[i,2]:.{precision}f}\n')
        f.write(f'\nPOINT_DATA {n_points}\n')
        f.write('SCALARS potential double 1\n')
        f.write('LOOKUP_TABLE default\n')
        for i in range(n_points):
            f.write(f'{scalars[i]:.{precision}f}\n')
