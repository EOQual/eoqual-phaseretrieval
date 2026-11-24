# Références

1. R. W. Gerchberg and W. O. Saxton, "A practical algorithm for the
   determination of phase from image and diffraction plane pictures,"
   *Optik*, 35, 237–246 (1972).
2. J. R. Fienup, "Reconstruction of an object from the modulus of its
   Fourier transform," *Optics Letters*, 3(1), 27–29 (1978).
3. R. J. Noll, "Zernike polynomials and atmospheric turbulence," *Journal
   of the Optical Society of America*, 66(3), 207–211 (1976).
4. J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., Roberts &
   Company Publishers (2005).
5. J. R. Fienup, "Phase retrieval algorithms: a comparison," *Applied
   Optics*, 21(15), 2758–2769 (1982).
6. M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior, and
   Conjugate Gradient Method for Large-Scale Bound-Constrained Minimization
   Problems," *SIAM Journal on Scientific Computing*, 21(1), 1–23 (1999).
   (algorithme *Trust Region Reflective*, utilisé par
   `scipy.optimize.least_squares(method="trf")`)
7. J. R. Fienup, "Phase-retrieval algorithms for a complicated optical
   system," *Applied Optics*, 32(10), 1737–1746 (1993).
8. R. A. Gonsalves, "Phase retrieval and diversity in adaptive optics,"
   *Optical Engineering*, 21(5), 829–832 (1982).
9. R. G. Paxman, T. J. Schulz, and J. R. Fienup, "Joint estimation of
   object and aberrations by using phase diversity," *Journal of the
   Optical Society of America A*, 9(7), 1072–1085 (1992).
10. M. G. Löfdahl and G. B. Scharmer, "Wavefront sensing and image
    restoration from focused and defocused solar images," *Astronomy and
    Astrophysics Supplement Series*, 107, 243–264 (1994).
11. P. J. Huber, "Robust Estimation of a Location Parameter," *Annals of
    Mathematical Statistics*, 35(1), 73–101 (1964).
12. R. P. Millane, "Phase retrieval in crystallography and optics,"
    *Journal of the Optical Society of America A*, 7(3), 394–411 (1990).
    (catalogue les ambiguïtés triviales de la récupération de phase, dont
    l'ambiguïté « image jumelle » de conjugaison/inversion utilisée en §3.7)
13. Ch. Latry, J.-M. Delvit, and C. Thiébaut, "Sensitivity analysis of
    phase diversity technique for high resolution earth observing
    telescopes," *Proc. SPIE* 10562, International Conference on Space
    Optics — ICSO 2016, 105623X (2017). doi:10.1117/12.2296051. (CNES —
    diversité de phase pour la mise au point en vol de Pléiades‑HR)
14. "Wave front sensing for next generation earth observation telescope,"
    *Proc. SPIE* 10562, ICSO 2016, 105625H (2017). doi:10.1117/12.2296162.
15. H. Ikoma, J.-M. Delvit, C. Latry, C. Thiébaut, and J.-M. Morel, "Phase
    Diversity: Estimation of an Optical Wavefront from Landscape Images,"
    *Applied Inverse Problems Conference*, Helsinki (2015). (CNES —
    diversité de phase myope, objet+aberration conjoints, sur imagerie de
    paysage plutôt que sur étoile)
16. J. Idier, L. M. Mugnier, and A. Blanc, "Statistical behavior of joint
    least-squares estimation in the phase diversity context," *IEEE
    Transactions on Image Processing*, 14(12), 2107–2116 (2005). (ONERA)
17. A. Blanc, L. M. Mugnier, and J. Idier, "Marginal estimation of
    aberrations and image restoration by use of phase diversity," *Journal
    of the Optical Society of America A*, 20(6), 1035–1045 (2003). (ONERA)
18. V. N. Mahajan, "Zernike annular polynomials for imaging systems with
    annular pupils," *Journal of the Optical Society of America*, 71(1),
    75–85 (1981). (polynômes orthogonaux pour pupille annulaire, §10)
19. G. Dai, "Modified Hartmann-Shack wavefront sensing and iterative
    wavefront reconstruction," *Optical Engineering*, 35(12), 3333–3340
    (1996). (§10, Gram-Schmidt)
20. D. J. Schroeder, *Astronomical Optics*, Academic Press (1987). (§10,
    Gram-Schmidt)
21. A. S. Fruchter and R. N. Hook, "Drizzle: A Method for the Linear
    Reconstruction of Undersampled Images," *Publications of the
    Astronomical Society of the Pacific*, 114, 144–152 (2002). (§5.0)
22. J. Anderson and I. R. King, "Toward High-Precision Astrometry with
    WFPC2: I. Deriving an Accurate Point-Spread Function," *Publications
    of the Astronomical Society of the Pacific*, 112, 1360–1382 (2000).
    (§5.0, méthode ePSF)

> Note de vérification : les références [13]–[17], [21]–[22] ont été
> identifiées par recherche bibliographique ciblée (titres, auteurs,
> revues/actes et années confirmés) plutôt que par lecture intégrale des
> articles. Les références [18]–[20] proviennent de la docstring d'un
> prototype antérieur de l'auteur (non conservé) et n'ont pas été re-vérifiées indépendamment.
> Dans tous les cas, les détails quantitatifs qu'elles contiennent restent
> à vérifier directement dans les textes avant citation dans un document
> externe à ce projet.
