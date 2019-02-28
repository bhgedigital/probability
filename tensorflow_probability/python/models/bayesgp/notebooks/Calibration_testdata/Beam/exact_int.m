function y=exact_int(u,Beta,L)


 C = (sin(Beta*L) + sinh(Beta*L))/(cos(Beta*L) + cosh(Beta*L));
  
 y = (sin(Beta.*u) - sinh(Beta.*u) - C*(cos(Beta.*u) - cosh(Beta.*u))).^2;