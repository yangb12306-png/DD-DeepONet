function out = Pipe_creat(dx1, dy1, dx2, dy2, dx3, dy3, U0, U1, data_id)
%
% Pipe_create.m
%
% Model exported on Dec 20 2025, 12:03 by COMSOL 6.1.0.252.

r1_size = [dx1, dy1];
r1_pos  = [0, 0];
r2_size = [dx2, dy2];
r2_pos  = [dx1, dy1 - dy2];
r3_size = [dx3, dy3];
r3_pos  = [dx1 + dx2 - dx3, dy1 - dy2 - dy3];

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

model.modelPath('D:\yangbo\2024-paper\paper\NS\comosl_batch_UZ\U_train');

model.component.create('comp1', true);

model.component('comp1').geom.create('geom1', 2);

model.component('comp1').mesh.create('mesh1');

model.component('comp1').physics.create('spf', 'LaminarFlow', 'geom1');

model.study.create('std1');
model.study('std1').create('stat', 'Stationary');
model.study('std1').feature('stat').setSolveFor('/physics/spf', true);

model.component('comp1').geom('geom1').lengthUnit('mm');
model.component('comp1').geom('geom1').create('r1', 'Rectangle');
model.component('comp1').geom('geom1').feature('r1').set('size', r1_size);
model.component('comp1').geom('geom1').run('r1');
model.component('comp1').geom('geom1').create('r2', 'Rectangle');
model.component('comp1').geom('geom1').feature('r2').set('size', r2_size);
model.component('comp1').geom('geom1').feature('r2').set('pos', r2_pos);
model.component('comp1').geom('geom1').run('r2');
model.component('comp1').geom('geom1').create('r3', 'Rectangle');
model.component('comp1').geom('geom1').feature('r3').set('size', r3_size);
model.component('comp1').geom('geom1').feature('r3').set('pos', r3_pos);
model.component('comp1').geom('geom1').run('r3');
model.component('comp1').geom('geom1').run('r3');
model.component('comp1').geom('geom1').create('uni1', 'Union');
model.component('comp1').geom('geom1').feature('uni1').selection('input').set({'r1' 'r2' 'r3'});
model.component('comp1').geom('geom1').feature('uni1').set('intbnd', false);
model.component('comp1').geom('geom1').run('uni1');
model.component('comp1').geom('geom1').run;

model.param.set('U0', U0);
model.param.set('U1', U1);
model.param.set('H',  dy1); 
model.component('comp1').func.create('an1', 'Analytic');
model.component('comp1').func('an1').set('funcname', 'inlet');
model.component('comp1').func('an1').set('expr', 'abs(U0*(1-(-2*(y/H)+1)^2)+U1*(-2*(y/H)+1)*(1-(-2*(y/H)+1)^2)-0.05*(1-(-2*(y/H)+1)^2)*(1-5*(-2*(y/H)+1)^2))');
model.component('comp1').func('an1').set('args', 'y');
model.component('comp1').func('an1').setIndex('argunit', 'mm', 0);

pin  = [0+1e-5 0-1e-5; 0 dy1];   % 入口附近一点（2D 用 [x;y]）
pout = [dx1+dx2-dx3+1e-5 dx1+dx2-dx3-1e-5; dy1-dy2-dy3 dy1-dy2];  % 出口附近一点（负 y 一侧给个点）
bnd_in  = mphselectcoords(model, 'geom1', pin,  'boundary');
bnd_out = mphselectcoords(model, 'geom1', pout, 'boundary');

model.component('comp1').physics('spf').feature('fp1').set('rho_mat', 'userdef');
model.component('comp1').physics('spf').feature('fp1').set('mu_mat', 'userdef');
model.component('comp1').physics('spf').create('inl1', 'InletBoundary', 1);
model.component('comp1').physics('spf').feature('inl1').selection.set([bnd_in]); % 4
% model.component('comp1').physics('spf').feature('inl1').set('U0in', U_in);
model.component('comp1').physics('spf').feature('inl1').set('U0in', 'inlet(y)');
model.component('comp1').physics('spf').create('out1', 'OutletBoundary', 1);
model.component('comp1').physics('spf').feature('out1').selection.set([bnd_out]); % 1
model.component('comp1').physics('spf').feature('out1').set('BoundaryCondition', 'FullyDevelopedFlow');

% model.component('comp1').mesh('mesh1').run;
% model.component('comp1').mesh('mesh1').autoMeshSize(3);
% model.component('comp1').mesh('mesh1').run;
model.component('comp1').mesh('mesh1').autoMeshSize(2);
model.component('comp1').mesh('mesh1').run;

model.sol.create('sol1');

model.component('comp1').mesh('mesh1').stat.selection.geom(2);
model.component('comp1').mesh('mesh1').stat.selection.set([1]);

model.sol('sol1').study('std1');

model.study('std1').feature('stat').set('notlistsolnum', 1);
model.study('std1').feature('stat').set('notsolnum', 'auto');
model.study('std1').feature('stat').set('listsolnum', 1);
model.study('std1').feature('stat').set('solnum', 'auto');

model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').feature('st1').set('study', 'std1');
model.sol('sol1').feature('st1').set('studystep', 'stat');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').feature('v1').set('control', 'stat');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').feature('s1').feature('aDef').set('cachepattern', true);
model.sol('sol1').feature('s1').create('fc1', 'FullyCoupled');
model.sol('sol1').feature('s1').feature('fc1').set('dtech', 'auto');
model.sol('sol1').feature('s1').feature('fc1').set('initstep', 0.01);
model.sol('sol1').feature('s1').feature('fc1').set('minstep', 1.0E-4);
model.sol('sol1').feature('s1').feature('fc1').set('maxiter', 100);
model.sol('sol1').feature('s1').create('d1', 'Direct');
model.sol('sol1').feature('s1').feature('d1').set('linsolver', 'pardiso');
model.sol('sol1').feature('s1').feature('d1').set('pivotperturb', 1.0E-13);
model.sol('sol1').feature('s1').feature('d1').label([native2unicode(hex2dec({'76' 'f4'}), 'unicode')  native2unicode(hex2dec({'63' 'a5'}), 'unicode')  native2unicode(hex2dec({'ff' '0c'}), 'unicode')  native2unicode(hex2dec({'6d' '41'}), 'unicode')  native2unicode(hex2dec({'4f' '53'}), 'unicode')  native2unicode(hex2dec({'6d' '41'}), 'unicode')  native2unicode(hex2dec({'52' 'a8'}), 'unicode')  native2unicode(hex2dec({'53' 'd8'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ' (spf)']);
model.sol('sol1').feature('s1').create('i1', 'Iterative');
model.sol('sol1').feature('s1').feature('i1').set('linsolver', 'gmres');
model.sol('sol1').feature('s1').feature('i1').set('prefuntype', 'left');
model.sol('sol1').feature('s1').feature('i1').set('itrestart', 50);
model.sol('sol1').feature('s1').feature('i1').set('rhob', 20);
model.sol('sol1').feature('s1').feature('i1').set('maxlinit', 1000);
model.sol('sol1').feature('s1').feature('i1').set('nlinnormuse', 'on');
model.sol('sol1').feature('s1').feature('i1').label(['AMG' native2unicode(hex2dec({'ff' '0c'}), 'unicode')  native2unicode(hex2dec({'6d' '41'}), 'unicode')  native2unicode(hex2dec({'4f' '53'}), 'unicode')  native2unicode(hex2dec({'6d' '41'}), 'unicode')  native2unicode(hex2dec({'52' 'a8'}), 'unicode')  native2unicode(hex2dec({'53' 'd8'}), 'unicode')  native2unicode(hex2dec({'91' 'cf'}), 'unicode') ' (spf)']);
model.sol('sol1').feature('s1').feature('i1').create('mg1', 'Multigrid');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('prefun', 'saamg');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('mgcycle', 'v');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('maxcoarsedof', 80000);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('strconn', 0.02);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('nullspace', 'constant');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('usesmooth', false);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('saamgcompwise', true);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('loweramg', true);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('compactaggregation', false);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').create('sc1', 'SCGS');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').feature('sc1').set('linesweeptype', 'ssor');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').feature('sc1').set('iter', 0);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').feature('sc1').set('scgsrelax', 0.7);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').feature('sc1').set('scgsmethod', 'lines');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').feature('sc1').set('scgsvertexrelax', 0.7);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').feature('sc1').set('relax', 0.5);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').feature('sc1').set('scgssolv', 'stored');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').feature('sc1').set('approxscgs', true);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').feature('sc1').set('scgsdirectmaxsize', 1000);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').create('sc1', 'SCGS');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').feature('sc1').set('linesweeptype', 'ssor');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').feature('sc1').set('iter', 1);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').feature('sc1').set('scgsrelax', 0.7);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').feature('sc1').set('scgsmethod', 'lines');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').feature('sc1').set('scgsvertexrelax', 0.7);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').feature('sc1').set('relax', 0.5);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').feature('sc1').set('scgssolv', 'stored');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').feature('sc1').set('approxscgs', true);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').feature('sc1').set('scgsdirectmaxsize', 1000);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('cs').create('d1', 'Direct');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('cs').feature('d1').set('linsolver', 'pardiso');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('cs').feature('d1').set('pivotperturb', 1.0E-13);
model.sol('sol1').feature('s1').feature('fc1').set('linsolver', 'd1');
model.sol('sol1').feature('s1').feature('fc1').set('dtech', 'auto');
model.sol('sol1').feature('s1').feature('fc1').set('initstep', 0.01);
model.sol('sol1').feature('s1').feature('fc1').set('minstep', 1.0E-4);
model.sol('sol1').feature('s1').feature('fc1').set('maxiter', 100);
model.sol('sol1').feature('s1').feature.remove('fcDef');
model.sol('sol1').attach('std1');

model.result.dataset('dset1').set('geom', 'geom1');
model.result.create('pg1', 'PlotGroup2D');
model.result('pg1').label([native2unicode(hex2dec({'90' '1f'}), 'unicode')  native2unicode(hex2dec({'5e' 'a6'}), 'unicode') ' (spf)']);
model.result('pg1').set('frametype', 'spatial');
model.result('pg1').set('defaultPlotID', 'ResultDefaults_SinglePhaseFlow/icom1/pdef1/pcond1/pg1');
model.result('pg1').feature.create('surf1', 'Surface');
model.result('pg1').feature('surf1').label([native2unicode(hex2dec({'88' '68'}), 'unicode')  native2unicode(hex2dec({'97' '62'}), 'unicode') ]);
model.result('pg1').feature('surf1').set('showsolutionparams', 'on');
model.result('pg1').feature('surf1').set('smooth', 'internal');
model.result('pg1').feature('surf1').set('showsolutionparams', 'on');
model.result('pg1').feature('surf1').set('data', 'parent');
model.result.create('pg2', 'PlotGroup2D');
model.result('pg2').label([native2unicode(hex2dec({'53' '8b'}), 'unicode')  native2unicode(hex2dec({'52' '9b'}), 'unicode') ' (spf)']);
model.result('pg2').set('frametype', 'spatial');
model.result('pg2').set('defaultPlotID', 'ResultDefaults_SinglePhaseFlow/icom1/pdef1/pcond1/pg2');
model.result('pg2').feature.create('con1', 'Contour');
model.result('pg2').feature('con1').label([native2unicode(hex2dec({'7b' '49'}), 'unicode')  native2unicode(hex2dec({'50' '3c'}), 'unicode')  native2unicode(hex2dec({'7e' 'bf'}), 'unicode') ]);
model.result('pg2').feature('con1').set('showsolutionparams', 'on');
model.result('pg2').feature('con1').set('expr', 'p');
model.result('pg2').feature('con1').set('number', 40);
model.result('pg2').feature('con1').set('levelrounding', false);
model.result('pg2').feature('con1').set('smooth', 'internal');
model.result('pg2').feature('con1').set('showsolutionparams', 'on');
model.result('pg2').feature('con1').set('data', 'parent');

model.sol('sol1').runAll;

model.result('pg1').run;

model.result.export.create('data1', 'Data');
model.result.export('data1').set('exporttype', 'vtu');
model.result.export('data1').set('descr', {});
model.result.export('data1').set('unit', {});
model.result.export('data1').set('expr', {'u' 'v' 'w'});
model.result.export('data1').setIndex('expr', 'sqrt(u*u+v*v)', 2);
model.result.export('data1').setIndex('descr', 'U', 2);
model.result.export('data1').setIndex('expr', 'p', 3);
vtu = ['C:\Users\DELL\Downloads\NS_revision\U_skewing\U_dataset\Pipe_ns_' num2str(data_id) '.vtu'];
model.result.export('data1').set('filename', vtu);
model.result.export('data1').run;

out = model;
end