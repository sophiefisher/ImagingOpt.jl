module ImagingOpt

    export PhysicsParams, RecoveryParams, ImagingParams, OptimizeParams, JobParams
    export prepare_incident, prepare_n2f_kernel, prepare_objects, prepare_geoms, prepare_blackbody, permittivity, prepare_surrogate, prepare_reconstruction, prepare_noises, prepare_fft_plans, 
prepare_noise_multiplier, prepare_weights
    export G, Gtranspose
    export nearfield, geoms_to_far, far_to_PSF, PSF_to_fftPSF, get_fftPSF, get_PSF, get_PSF_freespace, get_fftPSF_freespace, get_far, get_fftPSF_from_far, make_image_noiseless, make_image, reconstruct_object, reconstruct_object_optim,  reconstruction_objective, reconstruction_objective_simplified, gradient_reconstruction_T, gradient_reconstruction_T_autodiff,  get_image_diff_flat
    export dB_dT, d2B_dT2, Hes, term1plusterm2_hes, build_hessian, jacobian_vp_undiff, jacobian_vp_autodiff, jacobian_vp_manual, get_dsur_dg_times_incident, dloss_dparams
    export arrarr_to_multi
    export get_params, print_params, get_α, compute_system_params, compute_obj_and_grad, run_opt, process_opt, design_singlefreq_lens, load_MIT_Tmap

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
    using Optim
    using LineSearches
    using Interpolations

    include("prepare.jl")
    include("utils.jl")
    include("forward.jl")
    include("pipeline.jl")
    include("optimize.jl")
    include("backward.jl")

end

