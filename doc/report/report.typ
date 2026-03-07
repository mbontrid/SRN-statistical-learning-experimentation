#import "@preview/lilaq:0.5.0" as lq


#lq.diagram(
  width: 4cm,
  height: 4cm,
  lq.colormesh(
    lq.linspace(-4, 4, num: 10),
    lq.linspace(-4, 4, num: 10),
    (x, y) => x * y,
    map: color.map.magma,
  ),
)
