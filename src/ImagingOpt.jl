module ImagingOpt

    export PhysicsParams, RecoveryParams, ImagingParams, OptimizeParams, JobParams
    export prepare_incident, prepare_n2f_kernel, prepare_objects, prepare_geoms, prepare_blackbody, permittivity, prepare_surrogate, prepare_reconstruction, prepare_noises
    export G, Gtranspose
    export nearfield, geoms_to_far, far_to_PSFs, PSFs_to_fftPSFs, get_fftPSF, get_PSF, make_image, reconstruct_object, reconstruction_objective, reconstruction_objective_simplified, gradient_reconstruction_T, gradient_reconstruction_T_autodiff,  get_image_diff_flat, dB_dT, d2B_dT2
    export Hes, term1plusterm2_hes, build_hessian, jacobian_vp_undiff, jacobian_vp_autodiff, jacobian_vp_manual, dloss_dparams
    export arrarr_to_multi
    export get_params, get_α, compute_system_params, run_opt_test, run_opt, run_deterministic_opt, run_stochastic_opt, process_opt, design_singlefreq_lens

    using FFTW
    using UUIDs
    using Distributed
    using LinearAlgebra
    import LinearAlgebra.mul!
    using LinearMaps
    using Zygote: gradient, @adjoint, @showgrad, @nograd
    using Zygote
    using Random: randperm, randsubseq
    using Random
    using FiniteDifferences: central_fdm
    using Printf
    using Statistics
    using Setfield
    using StructTypes
    using Glob
    using JSON3
    using Dates
    using Optimisers
    using JLD2
    using FLoops
    using NaturalSort
    using Images
    using ShiftedArrays
    using Augmentor
    using TestImages
    using MAT
    using FastChebInterp
    using QuadratureRules
    using Profile
    using StatProfilerHTML
    using ProfileVega
    using WavePropagation
    using ChainRulesCore
    using StaticArrays
    using DelimitedFiles
    using JuMP
    using NLopt
    using ThreadsX
    using ChainRules
    using ChainRules: RuleConfig, HasReverseMode, rrule_via_ad, ProjectTo, NoTangent, unthunk
    using IterativeSolvers
    using PyPlot
    using Dates
    using LaTeXStrings
    using ImageQualityIndexes

    include("prepare.jl")
    include("utils.jl")
    include("forward.jl")
    include("pipeline.jl")
    include("optimize.jl")

end

