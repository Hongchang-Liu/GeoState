# Third-party notices

## VGGT-Omega

GeoState depends on VGGT-Omega from Meta Research. Source code and pretrained
weights are obtained separately from the official project and remain subject
to the upstream FAIR Noncommercial Research License and Acceptable Use Policy.
The primary release is pinned to commit
`39a0cb8af88554f15ddcb5354cd52bde588fa014` and does not vendor VGGT-Omega.

## Datasets

DL3DV, CO3D, RealEstate10K, TartanAir, Waymo Open Dataset, KITTI, and nuScenes
remain subject to their respective terms. The released manifests contain
dataset identifiers. The twelve example images in `examples/` are
derived from the Waymo Open Dataset and remain subject to its
[dataset terms](https://waymo.com/open/terms/); they are not covered by the
GeoState MIT license. The images come from Waymo training
segment `4392459808686681511_5006_200_5026_200`, with anchor timestamp
`1507242010391705` and history offsets of -0.6, -0.4, -0.2, and 0 seconds.
They are undistorted, balanced-512 images saved as lossless PNGs from the
prepared JPEG pixels.

## Project license

The GeoState source code in this repository is released under the MIT License;
see `LICENSE`. This license does not cover separately obtained VGGT-Omega code
or weights, dataset content, or third-party packages.
