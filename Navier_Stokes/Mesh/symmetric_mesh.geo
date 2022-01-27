// Define mesh, run with the command "gmsh symmetric_mesh.geo -2"

DefineConstant[ lc  = {1}];
DefineConstant[ lc1  = {0.1}];
DefineConstant[ radius  = {0.5}];

// Circle points
Point(1) = {0, 0, 0, lc1};
Point(2) = {-radius, 0, 0, lc1};
Point(3) = {0, radius, 0, lc1};

// Circle lines
Circle(10) = {3, 1, 2};

// Rectangle points
Point(4) = {-5, 5, 0, lc};
Point(5) = {-5, 0, 0, lc};
Point(6) = {15, 0, 0, lc};
Point(7) = {15, 5, 0, lc};

// Rectangle lines
Line(7) = {6, 7};
Line(8) = {7, 4};
Line(9) = {4, 5};

// Perform symmetry
Symmetry {0, 1, 0, 0} { Duplicata { Curve{9}; Curve{8}; Curve{7}; } }
Symmetry {1, 0, 0, 0.0} { Duplicata { Curve{10}; } }
Symmetry {0, 1, 0, 0.0} { Duplicata { Curve{14}; } }
Symmetry {-1, 0, 0, 0.0} { Duplicata { Curve{15}; } }
Line(17) = {5, 2};
Line(18) = {15, 6};
Curve Loop(1) = {8, 9, 17, -10, 14, 18, 7};
Plane Surface(1) = {1};
Symmetry {0, 1, 0, 0} {
  Duplicata { Surface{1}; }
}

// Remove central lines
Recursive Delete {
  Curve{17}; Curve{18}; 
}

// Create Physical curves
Physical Curve(10) = {11, 9};
Physical Curve(12) = {8, 12};
Physical Curve(11) = {7, 13};
Physical Curve(13) = {10, 16, 14, 15};
Physical Surface(1) = {1, 19};
