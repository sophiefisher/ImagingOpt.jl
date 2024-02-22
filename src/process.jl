function process_opt(presicion, parallel, opt_date, opt_id, pname)
    directory = "ImagingOpt.jl/optdata/$opt_id"
    
    MIT_reconstructions_directory = "ImagingOpt.jl/optdata/$opt_id/MIT_reconstructions"
    if ! isdir(MIT_reconstructions_directory)
        Base.Filesystem.mkdir(MIT_reconstructions_directory)
    end
    
    random_reconstructions_directory = "ImagingOpt.jl/optdata/$opt_id/random_reconstructions"
    if ! isdir(random_reconstructions_directory)
        Base.Filesystem.mkdir(random_reconstructions_directory)
    end
    
    PSFs_directory = "ImagingOpt.jl/optdata/$opt_id/more_PSFs"
    if ! isdir(PSFs_directory)
        Base.Filesystem.mkdir(PSFs_directory)
    end
    
    params = get_params("$(pname)_$(opt_date)", presicion, directory)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp
    surrogates, freqs = prepare_surrogate(pp)
    Tinit_flat = prepare_reconstruction(recp, imgp)
    plan_nearfar, plan_PSF = prepare_fft_plans(pp, imgp)
    weights = prepare_weights(pp)
    iqi = SSIM(KernelFactors.gaussian(1.5, 11), (1,1,1)) #standard parameters for SSIM
    if imgp.differentiate_noise
        noise_multiplier = 0
    else
        noise_multiplier = prepare_noise_multiplier(pp, imgp, surrogates, freqs, weights, plan_nearfar, plan_PSF, parallel)
    end
    
    Tmaps_random = prepare_objects(imgp, pp) #assuming random Tmaps
    noises_random = prepare_noises(imgp)
    
    Tmap_MIT = load_MIT_Tmap(imgp.objL, (imgp.lbT + imgp.ubT)/2, imgp.lbT + (imgp.ubT - imgp.lbT)*(3/4) )
    Tmaps_MIT = [ Tmap_MIT ]
    noises_MIT = [ imgp.noise_level .* randn(imgp.imgL, imgp.imgL); ]
    
    ################################# plot objective, alpha, and noise values throughout opt #################################
    println("######################### plotting objective and alpha values #########################")
    println()
    
    file_save_objective_vals = "$directory/objective_vals_$opt_date.csv"
    objdata = readdlm(file_save_objective_vals,',')
    figure(figsize=(18,10))
    suptitle(L"\mathrm{objective \  data } ,  \langle \frac{|| T - T_{est} ||^2}{  || T ||^2} \rangle_{T}")
    subplot(2,2,1)
    plot(objdata,".-")
    xlabel("iteration")
    title("objective values")

    subplot(2,2,2)
    semilogy(objdata,".-")
    xlabel("iteration")
    title("objective values (semilog plot)")

    file_save_best_objective_vals = "$directory/best_objective_vals_$opt_date.csv"
    objdata_best = readdlm(file_save_best_objective_vals,',')
    subplot(2,2,3)
    plot(objdata_best,".-",color="orange")
    xlabel("iteration")
    title("best objective values")

    subplot(2,2,4)
    semilogy(objdata_best,".-",color="orange")
    xlabel("iteration")
    title("best objective values (semilog plot)")
    
    tight_layout()
    savefig("$directory/objective_vals_$opt_date.png")
    
    if optp.optimize_alpha
        file_save_alpha_vals = "$directory/alpha_vals_$opt_date.csv"   
        alpha_vals = readdlm(file_save_alpha_vals,',')
        
        figure(figsize=(18,10))
        suptitle(L"\alpha \ \mathrm{values }")
        subplot(2,2,1)
        plot(alpha_vals,".-")
        xlabel("iteration")
        title("alpha values")

        subplot(2,2,2)
        semilogy(alpha_vals,".-")
        xlabel("iteration")
        title("alpha values (semilog plot)")
        
        file_save_best_alpha_vals = "$directory/best_alpha_vals_$opt_date.csv"   
        best_alpha_vals = readdlm(file_save_best_alpha_vals,',')
        subplot(2,2,3)
        plot(best_alpha_vals,".-",color="orange")
        xlabel("iteration")
        title("best alpha values")

        subplot(2,2,4)
        semilogy(best_alpha_vals,".-",color="orange")
        xlabel("iteration")
        title("best alpha values (semilog plot)")
        
        tight_layout()
        savefig("$directory/alpha_vals_$opt_date.png")
    end
    
    file_relative_noise_levels = "$directory/relative_noise_levels_$opt_date.csv"
    relative_noise_levels = readdlm(file_relative_noise_levels,',')
    figure(figsize=(12,6))
    title("noise levels (relative to mean image) throughout optimization")
    for obji = 1:imgp.objN
        plot(relative_noise_levels[:,obji],".-")
    end
    ylabel("% of mean image value")
    xlabel("iteration")
    legend(["$(obji)" for obji = 1:imgp.objN])
    tight_layout()
    savefig("$directory/relative_noise_levels_$opt_date.png")
    
    ################################# INITIAL geoms, alpha, and PSFs #################################
    #TO DO: if starting from random metasurface, save and then reload geoms here
    println("######################### plotting initial PSFs, reconstructions #########################")
    println()
    
    geoms_init = prepare_geoms(params)
    
    if parallel
        PSFs_init = ThreadsX.map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, parallel),1:pp.orderfreq+1)
    else
        PSFs_init = map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, parallel),1:pp.orderfreq+1)
    end
    
    if parallel
        fftPSFs_init = ThreadsX.map(iF-> PSF_to_fftPSF(PSFs_init[iF], plan_PSF),1:pp.orderfreq+1)
    else
        fftPSFs_init = map(iF-> PSF_to_fftPSF(PSFs_init[iF], plan_PSF),1:pp.orderfreq+1)
    end
    
    #=
    if parallel
        fftPSFs_init = ThreadsX.map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    else
        fftPSFs_init = map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    end
    =#
    α_init = optp.αinit
    
    #plot initial PSFs
    plot_PSFs(opt_date, directory, params, freqs, PSFs_init, parallel, "initial", 1, "different_linear")
    plot_PSFs(opt_date, directory, params, freqs, PSFs_init, parallel, "initial", 1, "same_linear")
    plot_PSFs(opt_date, directory, params, freqs, PSFs_init, parallel, "initial", 1, "same_log")
    
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_init, parallel, "initial",2, "same_linear")
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_init, parallel, "initial",4, "same_linear")
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_init, parallel, "initial",8, "same_linear")
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_init, parallel, "initial",16, "same_linear")
    
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_init, parallel, "initial",2, "same_log")
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_init, parallel, "initial",4, "same_log")
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_init, parallel, "initial",8, "same_log")
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_init, parallel, "initial",16, "same_log")
    
    #save initial random reconstruction
    plot_reconstruction(opt_date, random_reconstructions_directory, params, freqs, Tinit_flat, Tmaps_random, noises_random, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs_init, α_init, parallel, iqi, "initial", "random")
    
    #save initial MIT reconstruction
    plot_reconstruction(opt_date, MIT_reconstructions_directory, params, freqs, Tinit_flat, Tmaps_MIT, noises_random, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs_init, α_init, parallel, iqi, "initial", "MIT")
    
    #reconstructions for different noise levels, random and MIT Tmaps
    plot_reconstruction_fixed_noise_levels(opt_date, random_reconstructions_directory, params, freqs, Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_init, α_init, parallel, iqi, "initial", "random")
    
    plot_reconstruction_fixed_noise_levels(opt_date, MIT_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_init, α_init, parallel, iqi, "initial", "MIT")
    
    #reconstruction with different alpha values, random and MIT Tmaps, imgp.noise_level noise and 10% noise
    plot_reconstruction_different_alpha_vals(opt_date, random_reconstructions_directory, params, freqs, Tinit_flat, Tmaps_random[1], imgp.noise_level, plan_nearfar, plan_PSF, weights, fftPSFs_init, [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], parallel, iqi, "initial", "random")
    
    plot_reconstruction_different_alpha_vals(opt_date, MIT_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, imgp.noise_level, plan_nearfar, plan_PSF, weights, fftPSFs_init, [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], parallel, iqi, "initial", "MIT")
    
    plot_reconstruction_different_alpha_vals(opt_date, random_reconstructions_directory, params, freqs, Tinit_flat, Tmaps_random[1], 0.10, plan_nearfar, plan_PSF, weights, fftPSFs_init, [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], parallel, iqi, "initial", "random")
    
    plot_reconstruction_different_alpha_vals(opt_date, MIT_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, 0.10, plan_nearfar, plan_PSF, weights, fftPSFs_init, [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], parallel, iqi, "initial", "MIT")
    
    ################################# OPTIMIZED geoms, alpha, and PSFs #################################
    println("######################### plotting optimized PSFs, reconstructions #########################")
    println()
    
    geoms_filename = "$directory/geoms_optimized_$opt_date.csv"
    geoms_optimized = reshape(readdlm(geoms_filename,','),pp.gridL, pp.gridL )
    
    if parallel
        PSFs_optimized = ThreadsX.map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms_optimized, plan_nearfar, parallel),1:pp.orderfreq+1)
    else
        PSFs_optimized = map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms_optimized, plan_nearfar, parallel),1:pp.orderfreq+1)
    end
    
    if parallel
        fftPSFs_optimized = ThreadsX.map(iF->PSF_to_fftPSF(PSFs_optimized[iF], plan_PSF),1:pp.orderfreq+1)
    else
        fftPSFs_optimized = map(iF->PSF_to_fftPSF(PSFs_optimized[iF], plan_PSF),1:pp.orderfreq+1)
    end
    
    #=
    if parallel
        fftPSFs_optimized = ThreadsX.map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_optimized, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    else
        fftPSFs_optimized = map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_optimized, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    end
    =#
    
    if optp.optimize_alpha
        α_optimized = best_alpha_vals[end]
    else
        α_optimized = optp.αinit
    end
    
    #plot optimized PSFs
    plot_PSFs(opt_date, directory, params, freqs, PSFs_optimized, parallel, "optimized",1,"different_linear")
    plot_PSFs(opt_date, directory, params, freqs, PSFs_optimized, parallel, "optimized",1,"same_linear")
    plot_PSFs(opt_date, directory, params, freqs, PSFs_optimized, parallel, "optimized",1,"same_log")
    
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_optimized, parallel, "optimized",2,"same_linear")
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_optimized, parallel, "optimized",4,"same_linear")
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_optimized, parallel, "optimized",8,"same_linear")
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_optimized, parallel, "optimized",16,"same_linear")
    
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_optimized, parallel, "optimized",2,"same_log")
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_optimized, parallel, "optimized",4,"same_log")
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_optimized, parallel, "optimized",8,"same_log")
    plot_PSFs(opt_date, PSFs_directory, params, freqs, PSFs_optimized, parallel, "optimized",16,"same_log")
   
    #save optimized random reconstruction
    plot_reconstruction(opt_date, random_reconstructions_directory, params, freqs, Tinit_flat, Tmaps_random, noises_random, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random")
    
    #save optimized MIT reconstruction
    plot_reconstruction(opt_date, MIT_reconstructions_directory, params, freqs, Tinit_flat, Tmaps_MIT, noises_random, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT")
    
    #reconstructions for different noise levels, random and MIT Tmaps (optimized alpha and alpha=0)
    plot_reconstruction_fixed_noise_levels(opt_date, random_reconstructions_directory, params, freqs, Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random")
    
    plot_reconstruction_fixed_noise_levels(opt_date, random_reconstructions_directory, params, freqs, Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, 0, parallel, iqi, "optimized", "random")
    
    plot_reconstruction_fixed_noise_levels(opt_date, MIT_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT")
    
    plot_reconstruction_fixed_noise_levels(opt_date, MIT_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, 0, parallel, iqi, "optimized", "MIT")
    
    ################################# plot geoms init and optimized side by side #################################
    println("######################### plotting geoms #########################")
    println()
    
    figure(figsize=(18,8))
    subplot(1,3,1)
    imshow( geoms_init , vmin = pp.lbwidth, vmax = pp.ubwidth)
    colorbar()
    title("initial metasurface \n parameters")
    subplot(1,3,2)
    imshow(geoms_optimized, vmin = pp.lbwidth, vmax = pp.ubwidth)
    colorbar()
    title("optimized metasurface \n parameters")
    subplot(1,3,3)
    imshow(geoms_optimized)
    colorbar()
    title("optimized metasurface \n parameters")
    savefig("$directory/geoms_optimized_$opt_date.png")
    
    figure(figsize=(32,10))
    subplot(1,3,1)
    imshow( geoms_init , vmin = pp.lbwidth, vmax = pp.ubwidth)
    colorbar()
    title("initial metasurface \n parameters")
    subplot(1,3,2)
    imshow(geoms_optimized, vmin = pp.lbwidth, vmax = pp.ubwidth)
    colorbar()
    title("optimized metasurface \n parameters")
    subplot(1,3,3)
    imshow(geoms_optimized)
    colorbar()
    title("optimized metasurface \n parameters")
    savefig("$directory/geoms_optimized_bigger_$opt_date.png")
    
    ################################# get transmissions for initial and optimized metasurface #################################
    println("######################### getting transmissions #########################")
    println()
    
    get_fftPSF_freespace_iF = iF->get_fftPSF_freespace(freqs[iF], surrogates[iF], pp, imgp, plan_nearfar, plan_PSF)
    if parallel
        fftPSFs_freespace = ThreadsX.map(get_fftPSF_freespace_iF,1:pp.orderfreq+1)
    else
        fftPSFs_freespace = map(get_fftPSF_freespace_iF,1:pp.orderfreq+1)
    end
    
    transmission_relative_to_no_lens_initial = get_transmission_relative_to_no_lens(pp, imgp, Tmaps_random[1], fftPSFs_freespace, fftPSFs_init, freqs, weights,  plan_nearfar, plan_PSF, parallel)
    
    dict_output_initial = Dict("transmission_relative_to_no_lens" => transmission_relative_to_no_lens_initial)
    #save output data in json file
    output_data_filename = "$(directory)/transmissions_initial.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output_initial)
    end
    
    transmission_relative_to_no_lens_optimized = get_transmission_relative_to_no_lens(pp, imgp, Tmaps_random[1], fftPSFs_freespace, fftPSFs_optimized, freqs, weights, plan_nearfar, plan_PSF, parallel)
    
    transmission_optimized_relative_to_initial = get_optimized_transmission_relative_to_initial(pp, imgp, Tmaps_random[1], fftPSFs_optimized, fftPSFs_init, freqs, weights, plan_nearfar, plan_PSF, parallel)
    
    dict_output_optimized = Dict("transmission_relative_to_no_lens" => transmission_relative_to_no_lens_optimized, "transmission_relative_to_initial" => transmission_optimized_relative_to_initial)
    #save output data in json file
    output_data_filename = "$(directory)/transmissions_optimized.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output_optimized)
    end
    
    ################################# figure plots #################################
    println("######################### plotting nice figures #########################")
    println()
    
    figure_plots_intial_directory = "ImagingOpt.jl/optdata/$opt_id/figure_plots_initial"
    if ! isdir(figure_plots_intial_directory)
        Base.Filesystem.mkdir( figure_plots_intial_directory )
    end
    plot_geoms_figure(geoms_init, figure_plots_intial_directory)
    plot_geoms_bigger_figure(geoms_init, figure_plots_intial_directory)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 1, "different_linear")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 1, "same_linear")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 1, "same_log")
    
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 1, "same_linear", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 1, "same_log", true)
    
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 2, "same_linear")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 2, "same_log")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 4, "same_linear")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 4, "same_log")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 8, "same_linear")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 8, "same_log")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 16, "same_linear")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 16, "same_log")
    
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 2, "same_linear", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 2, "same_log", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 4, "same_linear", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 4, "same_log", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 8, "same_linear", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 8, "same_log", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 16, "same_linear", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_init, parallel, figure_plots_intial_directory, 16, "same_log", true)
    
    plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmap_MIT, α_init, fftPSFs_init, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, figure_plots_intial_directory, "MIT")
    plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmaps_random[1], α_init, fftPSFs_init, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, figure_plots_intial_directory, "random")
    
    
    figure_plots_optimized_directory = "ImagingOpt.jl/optdata/$opt_id/figure_plots_optimized"
    if ! isdir(figure_plots_optimized_directory)
        Base.Filesystem.mkdir( figure_plots_optimized_directory )
    end
    plot_geoms_figure(geoms_optimized, figure_plots_optimized_directory)
    plot_geoms_bigger_figure(geoms_optimized, figure_plots_optimized_directory)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,1,"different_linear")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,1,"same_linear")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,1,"same_log")
    
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,1,"same_linear", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,1,"same_log", true)
    
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,2,"same_linear")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,2,"same_log")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,4,"same_linear")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,4,"same_log")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,8,"same_linear")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,8,"same_log")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,16,"same_linear")
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,16,"same_log")
    
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,2,"same_linear", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,2,"same_log", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,4,"same_linear", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,4,"same_log", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,8,"same_linear", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,8,"same_log", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,16,"same_linear", true)
    plot_PSFs_figure(pp, imgp, freqs, PSFs_optimized, parallel, figure_plots_optimized_directory,16,"same_log", true)
    
    plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmap_MIT, α_optimized, fftPSFs_optimized, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, figure_plots_optimized_directory, "MIT")
    plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmaps_random[1], α_optimized, fftPSFs_optimized, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, figure_plots_optimized_directory, "random")
    
    ################################# reconstructions on different temperature ranges #################################
    println("######################### plotting reconstructions for different temperature ranges #########################")
    println()
    
    different_T_reconstructions_directory = "ImagingOpt.jl/optdata/$opt_id/different_temperature_reconstructions"
    if ! isdir(different_T_reconstructions_directory)
        Base.Filesystem.mkdir(different_T_reconstructions_directory)
    end
    
    bg = (imgp.lbT + imgp.ubT)/2
    logo = imgp.lbT + (imgp.ubT - imgp.lbT)*(1/4)
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = (imgp.lbT + imgp.ubT)/2
    logo = imgp.ubT 
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = (imgp.lbT + imgp.ubT)/2
    logo = imgp.lbT 
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = imgp.lbT 
    logo = imgp.ubT 
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = (imgp.lbT + imgp.ubT)/2 - 1
    logo = (imgp.lbT + imgp.ubT)/2 + 1
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 263.15
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 280
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 296
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 300
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 310
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 443.15
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 623.15
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 263.15
    logo = 623.15
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 280
    logo = 310
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    lbT = 263.15
    ubT = 623.15
    Tmap_random = rand(lbT:eps():ubT,imgp.objL, imgp.objL)
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_random, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random_$(lbT)_$(ubT)")
    
    lbT = 280
    ubT = 310
    Tmap_random = rand(lbT:eps():ubT,imgp.objL, imgp.objL)
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_random, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random_$(lbT)_$(ubT)")
    
    lbT = 294
    ubT = 296
    Tmap_random = rand(lbT:eps():ubT,imgp.objL, imgp.objL)
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_random, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random_$(lbT)_$(ubT)")
    
    lbT = (imgp.lbT + imgp.ubT)/2 - 1
    ubT = (imgp.lbT + imgp.ubT)/2 + 1
    Tmap_random = rand(lbT:eps():ubT,imgp.objL, imgp.objL)
    plot_reconstruction_fixed_noise_levels(opt_date, different_T_reconstructions_directory, params, freqs, Tinit_flat, Tmap_random, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random_$(lbT)_$(ubT)")
    
    ################################# calculating strehl ratios #################################
    println("######################### calculating strehl ratios #########################")
    println()
    
    strehl_ratios_directory = "ImagingOpt.jl/optdata/$opt_id/strehl_ratios"
    if ! isdir(strehl_ratios_directory)
        Base.Filesystem.mkdir(strehl_ratios_directory)
    end
    
    psfL = imgp.objL + imgp.imgL
    middle = div(psfL,2)
    
    if parallel
        PSFs_diff = ThreadsX.map(iF->get_PSF_diffraction_limited(freqs[iF], pp, imgp, plan_nearfar),1:pp.orderfreq+1)
    else
        PSFs_diff = map(iF->get_PSF_diffraction_limited(freqs[iF], pp, imgp, plan_nearfar),1:pp.orderfreq+1)
    end
    
    plot_PSFs(opt_date, strehl_ratios_directory, params, freqs, PSFs_diff, parallel, "ideal",1,"different_linear")
    plot_PSFs(opt_date, strehl_ratios_directory, params, freqs, PSFs_diff, parallel, "ideal",1,"same_linear")
    plot_PSFs(opt_date, strehl_ratios_directory, params, freqs, PSFs_diff, parallel, "ideal",1,"same_log")

    max_PSFs_diff = maximum.(PSFs_diff)
    max_PSFs_init = maximum.(PSFs_init)
    max_PSFs_optimized = maximum.(PSFs_optimized)
    
    mid_PSFs_diff = [(PSFs_diff[iF][middle,middle] + PSFs_diff[iF][middle+1,middle] + PSFs_diff[iF][middle,middle+1] + PSFs_diff[iF][middle+1,middle+1])/4 for iF in 1:pp.orderfreq + 1]
    mid_PSFs_init = [(PSFs_init[iF][middle,middle] + PSFs_init[iF][middle+1,middle] + PSFs_init[iF][middle,middle+1] + PSFs_init[iF][middle+1,middle+1])/4 for iF in 1:pp.orderfreq + 1]
    mid_PSFs_optimized = [(PSFs_optimized[iF][middle,middle] + PSFs_optimized[iF][middle+1,middle] + PSFs_optimized[iF][middle,middle+1] + PSFs_optimized[iF][middle+1,middle+1])/4 for iF in 1:pp.orderfreq + 1]
    
    absolute_strehl_ratios_init_max = max_PSFs_init ./ max_PSFs_diff
    absolute_strehl_ratios_optimized_max = max_PSFs_optimized ./ max_PSFs_diff 
    
    absolute_strehl_ratios_init_mid = mid_PSFs_init ./ mid_PSFs_diff
    absolute_strehl_ratios_optimized_mid = mid_PSFs_optimized ./ mid_PSFs_diff
    
    figure(figsize=(18,10))
    subplot(2,2,1)
    plot(1:length(freqs), absolute_strehl_ratios_init_max,".-")
    plot(1:length(freqs), absolute_strehl_ratios_optimized_max,".-")
    xticks(1:length(freqs), round.(freqs,digits=3), rotation=55 )
    xlabel("frequency")
    legend(["initial geometry","optimized geometry"])
    title("absolute strehl ratio, max val")
    
    subplot(2,2,2)
    semilogy(1:length(freqs), absolute_strehl_ratios_init_max,".-")
    semilogy(1:length(freqs), absolute_strehl_ratios_optimized_max,".-")
    xticks(1:length(freqs), round.(freqs,digits=3), rotation=55 )
    xlabel("frequency")
    legend(["initial geometry","optimized geometry"])
    title("absolute strehl ratio, max val (semilog plot)")
    
    subplot(2,2,3)
    plot(1:length(freqs), absolute_strehl_ratios_init_mid,".-")
    plot(1:length(freqs), absolute_strehl_ratios_optimized_mid,".-")
    xticks(1:length(freqs), round.(freqs,digits=3), rotation=55 )
    xlabel("frequency")
    legend(["initial geometry","optimized geometry"])
    title("absolute strehl ratio, middle val")
    
    subplot(2,2,4)
    semilogy(1:length(freqs), absolute_strehl_ratios_init_mid,".-")
    semilogy(1:length(freqs), absolute_strehl_ratios_optimized_mid,".-")
    xticks(1:length(freqs), round.(freqs,digits=3), rotation=55 )
    xlabel("frequency")
    legend(["initial geometry","optimized geometry"])
    title("absolute strehl ratio, middle val (semilog plot)")
    
    tight_layout()
    savefig("$(strehl_ratios_directory)/absolute_strehl_ratios_$(opt_date).png")
    
    writedlm("$(strehl_ratios_directory)/absolute_strehl_ratios_init_max",  absolute_strehl_ratios_init_max,',')
    writedlm("$(strehl_ratios_directory)/absolute_strehl_ratios_optimized_max",  absolute_strehl_ratios_optimized_max,',')
    
    writedlm("$(strehl_ratios_directory)/absolute_strehl_ratios_init_mid",  absolute_strehl_ratios_init_mid,',')
    writedlm("$(strehl_ratios_directory)/absolute_strehl_ratios_optimized_mid",  absolute_strehl_ratios_optimized_mid,',')
    
    #midline PSF figures
    plot_PSF_midlines(opt_date, strehl_ratios_directory, params, freqs, PSFs_diff, PSFs_init, "initial", "different_linear")
    plot_PSF_midlines(opt_date, strehl_ratios_directory, params, freqs, PSFs_diff, PSFs_init, "initial", "same_linear")
    plot_PSF_midlines(opt_date, strehl_ratios_directory, params, freqs, PSFs_diff, PSFs_init, "initial", "same_log")
    
    plot_PSF_midlines(opt_date, strehl_ratios_directory, params, freqs, PSFs_diff, PSFs_optimized, "optimized", "different_linear")
    plot_PSF_midlines(opt_date, strehl_ratios_directory, params, freqs, PSFs_diff, PSFs_optimized, "optimized", "same_linear")
    plot_PSF_midlines(opt_date, strehl_ratios_directory, params, freqs, PSFs_diff, PSFs_optimized, "optimized", "same_log")
    
    
    #compute diffraction limits
    mid_lines_diff = [PSFs_diff[i][middle, :] for i in 1:pp.orderfreq+1]
    peak_FWHM_pixel_sizes = Matrix{Float64}(undef, 1, pp.orderfreq+1)
    peak_FWHM_widths_μm = Matrix{Float64}(undef, 1, pp.orderfreq+1)
    diffraction_limit_calculations = Matrix{Float64}(undef, 1, pp.orderfreq+1)
    for i = 1:pp.orderfreq+1
        pks, vals = Peaks.findmaxima(mid_lines_diff[i])
        pks, proms = peakproms(pks, mid_lines_diff[i])
        pks, widths, _, _ = peakwidths(pks, mid_lines_diff[i], proms)
        idx = argmax(vals)
        width = widths[idx]
        peak_FWHM_pixel_sizes[i] = width
        peak_FWHM_widths_μm[i] = width * pp.wavcen * pp.cellL * imgp.binL
        diffraction_limit_calculations[i] = (pp.wavcen / freqs[i]) / (2*sin(atan( pp.gridL * pp.cellL / (2 * pp.F) )) )
    end

    data = [peak_FWHM_pixel_sizes; peak_FWHM_widths_μm; diffraction_limit_calculations]
    header = ["FWHM pixel size", "FWHM width (microns)", "calculated diff lim"]
    writedlm("$(strehl_ratios_directory)/diffraction_limits.csv",[header data],',')
    
    
    ################################# reconstructions for noisy PSFs #################################
    println("######################### plotting reconstructions for noisy PSFs #########################")
    println()
    
    noisy_PSFs_reconstructions_directory = "ImagingOpt.jl/optdata/$opt_id/noisy_PSFs_reconstructions"
    if ! isdir(noisy_PSFs_reconstructions_directory)
        Base.Filesystem.mkdir(noisy_PSFs_reconstructions_directory)
    end
    
    #initial
    plot_reconstruction_fixed_noise_levels_noisy_PSFs(opt_date, noisy_PSFs_reconstructions_directory, params, freqs, Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, PSFs_init, α_init, parallel, iqi, "initial", "random")
    
    plot_reconstruction_fixed_noise_levels_noisy_PSFs(opt_date, noisy_PSFs_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, PSFs_init, α_init, parallel, iqi, "initial", "MIT")
    
    #optimized
    plot_reconstruction_fixed_noise_levels_noisy_PSFs(opt_date, noisy_PSFs_reconstructions_directory, params, freqs, Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, PSFs_optimized, α_optimized, parallel, iqi, "optimized", "random")
    
    plot_reconstruction_fixed_noise_levels_noisy_PSFs(opt_date, noisy_PSFs_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, PSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT")
    
    ################################# reconstructions for interpolated PSFs #################################
    println("######################### plotting reconstructions for interpolated PSFs #########################")
    println()
    
    interpolated_PSFs_reconstructions_directory = "ImagingOpt.jl/optdata/$opt_id/interpolated_PSFs_reconstructions"
    if ! isdir(interpolated_PSFs_reconstructions_directory)
        Base.Filesystem.mkdir(interpolated_PSFs_reconstructions_directory)
    end
    
    interpolated_PSFs_optimized_reconstructions_directory = "ImagingOpt.jl/optdata/$opt_id/interpolated_PSFs_reconstructions/optimized"
    if ! isdir(interpolated_PSFs_optimized_reconstructions_directory)
        Base.Filesystem.mkdir(interpolated_PSFs_optimized_reconstructions_directory)
    end
    
    interpolated_PSFs_initial_reconstructions_directory = "ImagingOpt.jl/optdata/$opt_id/interpolated_PSFs_reconstructions/initial"
    if ! isdir(interpolated_PSFs_initial_reconstructions_directory)
        Base.Filesystem.mkdir(interpolated_PSFs_initial_reconstructions_directory)
    end
    
    #initial
    plot_reconstruction_fixed_noise_levels_interpolated_PSFs(opt_date, interpolated_PSFs_initial_reconstructions_directory, params, PSFs_init, freqs, [1; 11; 21], Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, α_init, parallel, iqi, "initial", "random")
    plot_reconstruction_fixed_noise_levels_interpolated_PSFs(opt_date, interpolated_PSFs_initial_reconstructions_directory, params, PSFs_init, freqs, [1; 11; 21], Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, α_init, parallel, iqi, "initial", "MIT")
    plot_reconstruction_fixed_noise_levels_interpolated_PSFs(opt_date, interpolated_PSFs_initial_reconstructions_directory, params, PSFs_init, freqs, [1; 8; 11; 14; 21], Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, α_init, parallel, iqi, "initial", "random")
    plot_reconstruction_fixed_noise_levels_interpolated_PSFs(opt_date, interpolated_PSFs_initial_reconstructions_directory, params, PSFs_init, freqs, [1; 8; 11; 14; 21], Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, α_init, parallel, iqi, "initial", "MIT")
    
    #optimized
    plot_reconstruction_fixed_noise_levels_interpolated_PSFs(opt_date, interpolated_PSFs_optimized_reconstructions_directory, params, PSFs_optimized, freqs, [1; 11; 21], Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, α_optimized, parallel, iqi, "optimized", "random")
    plot_reconstruction_fixed_noise_levels_interpolated_PSFs(opt_date, interpolated_PSFs_optimized_reconstructions_directory, params, PSFs_optimized, freqs, [1; 11; 21], Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, α_optimized, parallel, iqi, "optimized", "MIT")
    plot_reconstruction_fixed_noise_levels_interpolated_PSFs(opt_date, interpolated_PSFs_optimized_reconstructions_directory, params, PSFs_optimized, freqs, [1; 8; 11; 14; 21], Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, α_optimized, parallel, iqi, "optimized", "random")
    plot_reconstruction_fixed_noise_levels_interpolated_PSFs(opt_date, interpolated_PSFs_optimized_reconstructions_directory, params, PSFs_optimized, freqs, [1; 8; 11; 14; 21], Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, α_optimized, parallel, iqi, "optimized", "MIT")
    
end

function plot_reconstruction(opt_date, directory, params, freqs, Tinit_flat, Tmaps, noises, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs, α, parallel, iqi, geoms_type, Tmap_type)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp

    num_Tmaps = length(Tmaps)
    
    figure(figsize=(18,3.5*num_Tmaps))
    suptitle("$(geoms_type) reconstruction")
    for obji = 1:num_Tmaps
        Tmap = Tmaps[obji]
        noise = noises[obji]
        B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)

        image_Tmap_grid = make_image(pp, imgp, imgp.differentiate_noise, B_Tmap_grid, fftPSFs, freqs, weights, noise, noise_multiplier, plan_nearfar, plan_PSF, parallel);
        Test_flat, _ = reconstruct_object(image_Tmap_grid, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel)
        Test = reshape(Test_flat, imgp.objL, imgp.objL)
        
        subplot(num_Tmaps, 5, obji*5 - 4)
        imshow(Tmap, vmin = imgp.lbT, vmax = imgp.ubT)
        colorbar()
        title(L"T(x,y) \ %$obji")
    
        subplot(num_Tmaps, 5, obji*5 - 3)
        imshow(Test, vmin = imgp.lbT, vmax = imgp.ubT)
        colorbar()
        title(L"T_{est}(x,y) \ %$obji")
    
        subplot(num_Tmaps, 5, obji*5 - 2 )
        imshow( (Test .- Tmap)./Tmap .* 100)
        colorbar()
        MSE = sum((Tmap .- Test).^2) / sum(Tmap.^2)
        title("% difference \n MSE = $(round(MSE, digits=8))")
        
        ssim_map = ImageQualityIndexes._ssim_map(iqi, Tmap, Test )
        subplot(num_Tmaps, 5, obji*5 - 1  )
        imshow( ssim_map )
        colorbar()
        title("SSIM \n mean = $( round(mean(ssim_map),digits=8)  )")
        
        subplot(num_Tmaps, 5, obji*5)
        imshow(image_Tmap_grid)
        colorbar()
        # calculate new noise level
        if imgp.differentiate_noise
            noise_level = imgp.noise_level
        else
            img_noiseless = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
            noise_level = imgp.noise_level * noise_multiplier / mean(img_noiseless)
        end
        title("image \n noise level is $( round(noise_level * 100,digits=4) )%")
    end
    tight_layout()
    savefig("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_$(opt_date).png")
end

function plot_reconstruction_fixed_noise_levels(opt_date, directory, params, freqs, Tinit_flat, Tmap, noise_levels, plan_nearfar, plan_PSF, weights, fftPSFs, α, parallel, iqi, geoms_type, Tmap_type)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp
    
    B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
    num_noise_levels = length(noise_levels)
    
    MSEs = Vector{Float64}(undef, length(noise_levels))
    
    figure(figsize=(18,3.5*num_noise_levels))
    suptitle("$(Tmap_type) reconstruction at fixed noise levels ($(geoms_type)). α = $(@sprintf "%.4e" α )")
    for i = 1:num_noise_levels
        noise_level = noise_levels[i]
        noise = noise_level .* randn(imgp.imgL, imgp.imgL)
        image_Tmap_grid = make_image(pp, imgp, true, B_Tmap_grid, fftPSFs, freqs, weights, noise, 0, plan_nearfar, plan_PSF, parallel);
        Test_flat, _ = reconstruct_object(image_Tmap_grid, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel)
        Test = reshape(Test_flat, imgp.objL, imgp.objL)
        
        subplot(num_noise_levels, 5, i*5 - 4)
        imshow(Tmap, vmin = minimum(Tmap), vmax = maximum(Tmap))
        colorbar()
        title(L"T(x,y)")
    
        subplot(num_noise_levels, 5, i*5 - 3)
        imshow(Test, vmin = minimum(Tmap), vmax = maximum(Tmap))
        colorbar()
        title(L"T_{est}(x,y)")
    
        subplot(num_noise_levels, 5, i*5 - 2 )
        imshow( (Test .- Tmap)./Tmap .* 100)
        colorbar()
        MSE = sum((Tmap .- Test).^2) / sum(Tmap.^2)
        MSEs[i] = MSE
        title("% difference \n MSE = $(round(MSE, digits=6))")
        
        ssim_map = ImageQualityIndexes._ssim_map(iqi, Tmap, Test )
        subplot(num_noise_levels, 5, i*5 - 1  )
        imshow( ssim_map )
        colorbar()
        title("SSIM \n mean = $( round(mean(ssim_map),digits=6)  )")
        
        subplot(num_noise_levels, 5, i*5)
        imshow(image_Tmap_grid)
        colorbar()
        title("image \n noise level is $(noise_level * 100)%")
    end
    tight_layout(rect=[0, 0, 1, 0.99])
    savefig("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_$(@sprintf "%.4e" α )_fixed_noises_$(opt_date).png")
    
    open("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_$(@sprintf "%.4e" α )_fixed_noises_MSEs_$(opt_date).csv", "w") do io
        writedlm(io, [noise_levels, MSEs],',')
    end
end


function plot_reconstruction_fixed_noise_levels_noisy_PSFs(opt_date, directory, params, freqs, Tinit_flat, Tmap, noise_levels, plan_nearfar, plan_PSF, weights, PSFs, α, parallel, iqi, geoms_type, Tmap_type)
    #assumes PSF noise is the same as image noise (relative to mean for both the image and the PSF)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp
    psfL = imgp.objL + imgp.imgL
    
    #compute true fftPSFs
    if parallel
        fftPSFs = ThreadsX.map(iF-> PSF_to_fftPSF(PSFs[iF], plan_PSF),1:pp.orderfreq+1)
    else
        fftPSFs = map(iF-> PSF_to_fftPSF(PSFs[iF], plan_PSF),1:pp.orderfreq+1)
    end
    
    B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
    num_noise_levels = length(noise_levels)
    
    MSEs = Vector{Float64}(undef, length(noise_levels))
    
    figure(figsize=(18,3.5*num_noise_levels))
    suptitle("$(Tmap_type) reconstruction at fixed noise levels with noisy PSFs ($(geoms_type)). α = $(@sprintf "%.4e" α )")
    for i = 1:num_noise_levels
        noise_level = noise_levels[i]
        noise = noise_level .* randn(imgp.imgL, imgp.imgL)
        
        #image gets formed with true PSFs
        image_Tmap_grid = make_image(pp, imgp, true, B_Tmap_grid, fftPSFs, freqs, weights, noise, 0, plan_nearfar, plan_PSF, parallel);
        
        #get noisy PSFs/fftPSFs
        noisy_PSFs = [PSFs[i] .+ (mean(PSFs[i]) .* noise_level .* randn(psfL, psfL)) for i in 1:pp.orderfreq+1]
        #compute true fftPSFs
        if parallel
            noisy_fftPSFs = ThreadsX.map(iF-> PSF_to_fftPSF(noisy_PSFs[iF], plan_PSF),1:pp.orderfreq+1)
        else
            noisy_fftPSFs = map(iF-> PSF_to_fftPSF(noisy_PSFs[iF], plan_PSF),1:pp.orderfreq+1)
        end
        
        #reconstruction is done with noisy PSFs
        Test_flat, _ = reconstruct_object(image_Tmap_grid, Tinit_flat, pp, imgp, optp, recp, noisy_fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel)
        Test = reshape(Test_flat, imgp.objL, imgp.objL)
        
        subplot(num_noise_levels, 5, i*5 - 4)
        imshow(Tmap, vmin = minimum(Tmap), vmax = maximum(Tmap))
        colorbar()
        title(L"T(x,y)")
    
        subplot(num_noise_levels, 5, i*5 - 3)
        imshow(Test, vmin = minimum(Tmap), vmax = maximum(Tmap))
        colorbar()
        title(L"T_{est}(x,y)")
    
        subplot(num_noise_levels, 5, i*5 - 2 )
        imshow( (Test .- Tmap)./Tmap .* 100)
        colorbar()
        MSE = sum((Tmap .- Test).^2) / sum(Tmap.^2)
        MSEs[i] = MSE
        title("% difference \n MSE = $(round(MSE, digits=6))")
        
        ssim_map = ImageQualityIndexes._ssim_map(iqi, Tmap, Test )
        subplot(num_noise_levels, 5, i*5 - 1  )
        imshow( ssim_map )
        colorbar()
        title("SSIM \n mean = $( round(mean(ssim_map),digits=6)  )")
        
        subplot(num_noise_levels, 5, i*5)
        imshow(image_Tmap_grid)
        colorbar()
        title("image \n noise level is $(noise_level * 100)%")
    end
    tight_layout(rect=[0, 0, 1, 0.99])
    savefig("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_$(@sprintf "%.4e" α )_fixed_noises_noisy_PSFs_$(opt_date).png")
    
    open("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_$(@sprintf "%.4e" α )_fixed_noises_noisy_PSFs_MSEs_$(opt_date).csv", "w") do io
        writedlm(io, [noise_levels, MSEs],',')
    end
end


function plot_reconstruction_fixed_noise_levels_interpolated_PSFs(opt_date, directory, params, PSFs, freqs, idxs_have, Tinit_flat, Tmap, noise_levels, plan_nearfar, plan_PSF, weights, α, parallel, iqi, geoms_type, Tmap_type)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp
    psfL = imgp.objL + imgp.imgL
    
    #compute true fftPSFs
    if parallel
        fftPSFs = ThreadsX.map(iF-> PSF_to_fftPSF(PSFs[iF], plan_PSF),1:pp.orderfreq+1)
    else
        fftPSFs = map(iF-> PSF_to_fftPSF(PSFs[iF], plan_PSF),1:pp.orderfreq+1)
    end
    
    freqs_have = freqs[idxs_have]
    PSFs_have = PSFs[idxs_have]

    itp = interpolate((freqs_have,), PSFs_have, Gridded(Linear()))
    PSFs_interpolated = [itp(freqs[i]) for i in 1:length(freqs)]
    
    #plot PSFs
    plot_PSFs(opt_date, directory, params, freqs, PSFs_interpolated, parallel, "$(geoms_type)_interpolated_$(join(idxs_have,"_ "))", 1, "different_linear")
    plot_PSFs(opt_date, directory, params, freqs, PSFs_interpolated, parallel, "$(geoms_type)_interpolated_$(join(idxs_have,"_ "))", 1, "same_linear")
    plot_PSFs(opt_date, directory, params, freqs, PSFs_interpolated, parallel, "$(geoms_type)_interpolated_$(join(idxs_have,"_ "))", 1, "same_log")
    
    #compute interpolated fftPSFs
    if parallel
        fftPSFs_interpolated = ThreadsX.map(iF-> PSF_to_fftPSF(PSFs_interpolated[iF], plan_PSF),1:pp.orderfreq+1)
    else
        fftPSFs_interpolated = map(iF-> PSF_to_fftPSF(PSFs_interpolated[iF], plan_PSF),1:pp.orderfreq+1)
    end
    
    B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
    num_noise_levels = length(noise_levels)
    
    MSEs = Vector{Float64}(undef, length(noise_levels))
    
    figure(figsize=(18,3.5*num_noise_levels))
    suptitle("$(Tmap_type) reconstruction at fixed noise levels with interpolated PSFs ($(geoms_type)). α = $(@sprintf "%.4e" α ) \n PSF indices used = $(join(idxs_have,", ")); freqs = $(join( round.(freqs_have,digits=3),", ")) ")
    for i = 1:num_noise_levels
        noise_level = noise_levels[i]
        noise = noise_level .* randn(imgp.imgL, imgp.imgL)
        
        #image formed with true fftPSFs
        image_Tmap_grid = make_image(pp, imgp, true, B_Tmap_grid, fftPSFs, freqs, weights, noise, 0, plan_nearfar, plan_PSF, parallel);
        
        #reconstruction done with interpolated fftPSFs
        Test_flat, _ = reconstruct_object(image_Tmap_grid, Tinit_flat, pp, imgp, optp, recp, fftPSFs_interpolated, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel)
        Test = reshape(Test_flat, imgp.objL, imgp.objL)
        
        subplot(num_noise_levels, 5, i*5 - 4)
        imshow(Tmap, vmin = minimum(Tmap), vmax = maximum(Tmap))
        colorbar()
        title(L"T(x,y)")
    
        subplot(num_noise_levels, 5, i*5 - 3)
        imshow(Test, vmin = minimum(Tmap), vmax = maximum(Tmap))
        colorbar()
        title(L"T_{est}(x,y)")
    
        subplot(num_noise_levels, 5, i*5 - 2 )
        imshow( (Test .- Tmap)./Tmap .* 100)
        colorbar()
        MSE = sum((Tmap .- Test).^2) / sum(Tmap.^2)
        MSEs[i] = MSE
        title("% difference \n MSE = $(round(MSE, digits=6))")
        
        ssim_map = ImageQualityIndexes._ssim_map(iqi, Tmap, Test )
        subplot(num_noise_levels, 5, i*5 - 1  )
        imshow( ssim_map )
        colorbar()
        title("SSIM \n mean = $( round(mean(ssim_map),digits=6)  )")
        
        subplot(num_noise_levels, 5, i*5)
        imshow(image_Tmap_grid)
        colorbar()
        title("image \n noise level is $(noise_level * 100)%")
    end
    tight_layout(rect=[0, 0, 1, 0.99])
    savefig("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_$(@sprintf "%.4e" α )_fixed_noises_interpolated_PSFs_$(join(idxs_have,"_"))_$(opt_date).png")
    
    open("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_$(@sprintf "%.4e" α )_fixed_noises_interpolated_PSFs_$(join(idxs_have,"_"))_MSEs_$(opt_date).csv", "w") do io
        writedlm(io, [noise_levels, MSEs],',')
    end
end



function plot_reconstruction_different_alpha_vals(opt_date, directory, params, freqs, Tinit_flat, Tmap, noise_level, plan_nearfar, plan_PSF, weights, fftPSFs, alpha_vals, parallel, iqi, geoms_type, Tmap_type)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp
    
    B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
    num_alpha_vals = length(alpha_vals)
    
    MSEs = Vector{Float64}(undef, num_alpha_vals)
    
    figure(figsize=(18,3.5*num_alpha_vals))
    suptitle("$(Tmap_type) reconstruction with different alpha values ($(geoms_type)). noise level = $(noise_level * 100)%")
    for i = 1:num_alpha_vals
        α = alpha_vals[i]
        noise = noise_level .* randn(imgp.imgL, imgp.imgL)
        image_Tmap_grid = make_image(pp, imgp, true, B_Tmap_grid, fftPSFs, freqs, weights, noise, 0, plan_nearfar, plan_PSF, parallel);
        Test_flat, _ = reconstruct_object(image_Tmap_grid, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel)
        Test = reshape(Test_flat, imgp.objL, imgp.objL)
        
        subplot(num_alpha_vals, 5, i*5 - 4)
        imshow(Tmap, vmin = minimum(Tmap), vmax = maximum(Tmap))
        colorbar()
        title(L"T(x,y)")
    
        subplot(num_alpha_vals, 5, i*5 - 3)
        imshow(Test, vmin = minimum(Tmap), vmax = maximum(Tmap))
        colorbar()
        title(L"T_{est}(x,y)")
    
        subplot(num_alpha_vals, 5, i*5 - 2 )
        imshow( (Test .- Tmap)./Tmap .* 100)
        colorbar()
        MSE = sum((Tmap .- Test).^2) / sum(Tmap.^2)
        MSEs[i] = MSE
        title("% difference \n MSE = $(round(MSE, digits=6))")
        
        ssim_map = ImageQualityIndexes._ssim_map(iqi, Tmap, Test )
        subplot(num_alpha_vals, 5, i*5 - 1  )
        imshow( ssim_map )
        colorbar()
        title("SSIM \n mean = $( round(mean(ssim_map),digits=6)  )")
        
        subplot(num_alpha_vals, 5, i*5)
        imshow(image_Tmap_grid)
        colorbar()
        title("image. α = $(@sprintf "%.4e" α )")
    end
    tight_layout(rect=[0, 0, 1, 0.99])
    savefig("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_noise_$(noise_level)_different_alpha_vals_$(opt_date).png")
    
    open("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_noise_$(noise_level)_different_alpha_vals_MSEs_$(opt_date).csv", "w") do io
        writedlm(io, [alpha_vals, MSEs],',')
    end
end

#TO DO: take PSFs as input so you don't have to recompute each time
function plot_PSFs(opt_date, directory, params, freqs, PSFs, parallel, geoms_type, cropfactor=1, scaling = "different_linear")
    pp = params.pp
    imgp = params.imgp
    psfL = imgp.objL + imgp.imgL
    
    PSF_function = iF->PSFs[iF][(psfL ÷ 2) - (psfL ÷ cropfactor ÷ 2) + 1 : (psfL ÷ 2) + (psfL ÷ cropfactor ÷ 2), (psfL ÷ 2) - (psfL ÷ cropfactor ÷ 2) + 1 : (psfL ÷ 2) + (psfL ÷ cropfactor ÷ 2) ]
    if parallel
        PSFs_cropped = ThreadsX.map(PSF_function,1:pp.orderfreq+1)
    else
        PSFs_cropped = map(PSF_function,1:pp.orderfreq+1)
    end
    maxval = maximum(maximum.(PSFs_cropped))
    minval = minimum(minimum.(PSFs_cropped))
    
    figure(figsize=(18,11))
    #assumes 21 PSFs
    for i = 1:21
        subplot(3,7,i)
        if scaling == "different_linear"
            imshow(PSFs_cropped[i])
        elseif scaling == "same_linear"
            imshow(PSFs_cropped[i], vmin = minval, vmax = maxval)
        elseif scaling == "same_log"
            imshow(PSFs_cropped[i], norm=matplotlib[:colors][:LogNorm](vmin = minval, vmax = maxval))
        end
        colorbar()
        title("ν = $(round(freqs[i],digits=3) )")
        axis("off")
    end
    tight_layout(rect=[0, 0, 1, 0.97])
    suptitle("$(geoms_type) PSFs. cropping = $(cropfactor); scaling = $(scaling).")
    if cropfactor==1
        filename = "$directory/PSFs_$(geoms_type)_scaling_$(scaling)_$opt_date.png"
    else
        filename = "$directory/PSFs_$(geoms_type)_cropped_$(cropfactor)_scaling_$(scaling)_$opt_date.png"
    end
    savefig(filename)
end
    


function plot_PSF_midlines(opt_date, directory, params, freqs, PSFs_diff, PSFs, geoms_type, scaling="different_linear")
    pp = params.pp
    imgp = params.imgp
    psfL = imgp.objL + imgp.imgL
    middle = div(psfL,2)
    
    mid_lines_diff = [PSFs_diff[i][middle, :] for i in 1:pp.orderfreq+1]
    mid_lines = [PSFs[i][middle, :] for i in 1:pp.orderfreq+1]
    
    maxval_diff = maximum(maximum.(mid_lines_diff))
    minval_diff = minimum(minimum.(mid_lines_diff))
    maxval_other = maximum(maximum.(mid_lines))
    minval_other = minimum(minimum.(mid_lines))
                    
    maxval = maximum([maxval_diff, maxval_other])
    minval = minimum([minval_diff, minval_other])
    
    xvals = range(-psfL/2 + 0.5, psfL/2 - 0.5, length = psfL) .* pp.cellL .* imgp.binL .* pp.wavcen
    
    figure(figsize=(18,10))
    suptitle("PSF midlines for ideal geometry (blue) and $(geoms_type) geometry (orange)")
    #assumes 21 PSFs
    for i = 1:21
        subplot(3,7,i)
        if scaling == "different_linear"
            plot(xvals, mid_lines_diff[i],".-")
            plot(xvals, mid_lines[i],".-")
        elseif scaling == "same_linear"
            plot(xvals, mid_lines_diff[i],".-")
            plot(xvals, mid_lines[i],".-")
            ylim([minval, maxval])
        elseif scaling == "same_log"
            semilogy(xvals, mid_lines_diff[i],".-")
            semilogy(xvals, mid_lines[i],".-")
            xlabel("μm")
            ylim([minval, maxval])
        end
        title("ν = $(round(freqs[i],digits=3) )")
        #legend(["ideal PSF","$(geoms_type) PSF"])
    end
    tight_layout(rect=[0, 0, 1, 0.99])
    filename = "$directory/PSFs_midlines_$(geoms_type)_scaling_$(scaling)_$opt_date.png"
    savefig(filename)
    
end
    
     

function get_transmission_relative_to_no_lens(pp, imgp, Tmap, fftPSFs_freespace, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
    B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
    image_Tmap_grid_freespace = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs_freespace, freqs, weights, plan_nearfar, plan_PSF, parallel)
    image_Tmap_grid_nonoise = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
    transmission = round(sum(image_Tmap_grid_nonoise) / sum(image_Tmap_grid_freespace) * 100,digits=4)
end
    
function get_optimized_transmission_relative_to_initial(pp, imgp, Tmap, fftPSFs_optimized, fftPSFs_init, freqs, weights, plan_nearfar, plan_PSF, parallel)
    B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
    image_Tmap_grid_optimized = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs_optimized, freqs, weights, plan_nearfar, plan_PSF, parallel)
    image_Tmap_grid_initial = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs_init, freqs, weights, plan_nearfar, plan_PSF, parallel)
    transmission = round(sum(image_Tmap_grid_optimized) / sum(image_Tmap_grid_initial) * 100,digits=4)
end

function plot_geoms_figure(geoms, directory)
    figure(figsize=(5,5))
    imshow(geoms, cmap="Greys_r")
    axis("off")
    savefig("$(directory)/geoms.png")
end

function plot_geoms_bigger_figure(geoms, directory)
    figure(figsize=(10,10))
    imshow(geoms, cmap="Greys_r")
    axis("off")
    savefig("$(directory)/geoms_bigger.png")
end

#TO DO: take PSFs as input so you don't have to recompute each time
function plot_PSFs_figure(pp, imgp, freqs, PSFs, parallel, directory, cropfactor=1, scaling = "different_linear",show_colorbar=false)
    #plot PSFs (make sure there are only 21 of them)
    psfL = imgp.objL + imgp.imgL
    
    PSF_function = iF->PSFs[iF][(psfL ÷ 2) - (psfL ÷ cropfactor ÷ 2) + 1 : (psfL ÷ 2) + (psfL ÷ cropfactor ÷ 2), (psfL ÷ 2) - (psfL ÷ cropfactor ÷ 2) + 1 : (psfL ÷ 2) + (psfL ÷ cropfactor ÷ 2) ]
    if parallel
        PSFs_cropped = ThreadsX.map(PSF_function,1:pp.orderfreq+1)
    else
        PSFs_cropped = map(PSF_function,1:pp.orderfreq+1)
    end
    maxval = maximum(maximum.(PSFs_cropped))
    minval = minimum(minimum.(PSFs_cropped))
    
    rc("font",  size=15)
    if show_colorbar
        fig = figure(figsize=(14,6.5))
    else
        fig = figure(figsize=(12,6))
    end
    for i = 1:21
        subplot(3,7,i)
        if scaling == "different_linear"
            imshow(PSFs_cropped[i])
        elseif scaling == "same_linear"
            imshow(PSFs_cropped[i], vmin = minval, vmax = maxval)
        elseif scaling == "same_log"
            imshow(PSFs_cropped[i], norm=matplotlib[:colors][:LogNorm](vmin = minval, vmax = maxval))
        end
        wavelength = pp.wavcen ./ freqs[i]
        title("λ = $(round(wavelength,digits=2) ) μm",fontsize=15)
        axis("off")
    end
    if show_colorbar
        fig.subplots_adjust(left=0.05, right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.13, 0.02, 0.74])
        colorbar(cax=cbar_ax)
    else
        tight_layout()    
    end
    
    if cropfactor==1
        filename = "$directory/PSFs_scaling_$(scaling)_colorbar_$(show_colorbar).png"
    else
        filename = "$directory/PSFs_cropped_$(cropfactor)_scaling_$(scaling)_colorbar_$(show_colorbar).png"
    end
    savefig(filename)
end

function plot_Tmap_figure(imgp, Tmap, directory, Tmap_type)
    figure(figsize=(4,4))
    imshow(Tmap, vmin = imgp.lbT, vmax = imgp.ubT, cmap="Reds")
    clb = colorbar()
    axis("off")
    clb.set_label("Temperature (°K)",fontsize=15)
    clb.ax.tick_params(labelsize=15)
    savefig("$(directory)/$(Tmap_type)_Tmap.png")
end

function plot_image_figure(pp, imgp, Tmap, fftPSFs, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, directory, Tmap_type, noise_model_type)
    #noise_model_type should either be "default" or a percentage value as a string like "0.02", "0.05", etc.
    B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
    
    if noise_model_type == "default"
        
        noise = imgp.noise_level .* randn(imgp.imgL, imgp.imgL);
        image_Tmap_grid = make_image(pp, imgp, imgp.differentiate_noise, B_Tmap_grid, fftPSFs, freqs, weights, noise, noise_multiplier, plan_nearfar, plan_PSF, parallel);
        if imgp.differentiate_noise
            noise_level = imgp.noise_level * 100
        else
            img_noiseless = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
            noise_level = imgp.noise_level * noise_multiplier / mean(img_noiseless) * 100
        end
        noise_model_type_string = "$(noise_model_type)_$(round(noise_level ,digits=4))"
    #otherwise, noise_model_type is equal to noise_level 
        
    else
        
        noise_level = parse(Float64,noise_model_type) / 100
        noise = noise_level .* randn(imgp.imgL, imgp.imgL);
        image_Tmap_grid = make_image(pp, imgp, true, B_Tmap_grid, fftPSFs, freqs, weights, noise, 0, plan_nearfar, plan_PSF, parallel)
        noise_model_type_string = "$(noise_model_type)"
        
    end
    
    figure(figsize=(4,4))
    imshow(image_Tmap_grid)
    #colorbar()
    axis("off")
    savefig("$(directory)/$(Tmap_type)_image_$(noise_model_type_string).png")
    image_Tmap_grid, noise_model_type
end

function plot_reconstruction_figure(pp, imgp, optp, recp, image_Tmap_grid, Tinit_flat, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, parallel, directory, Tmap_type, noise_model_type)
    Test_flat, _ = reconstruct_object(image_Tmap_grid, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel)
    Test = reshape(Test_flat, imgp.objL, imgp.objL)
    figure(figsize=(4,4))
    imshow(Test, vmin = imgp.lbT, vmax = imgp.ubT, cmap="Reds")
    clb = colorbar()
    axis("off")
    clb.set_label("Temperature (°K)",fontsize=15)
    clb.ax.tick_params(labelsize=15)
    savefig("$(directory)/$(Tmap_type)_Tmap_reconstructed_$(noise_model_type).png")
    Test
end

function plot_percent_error_figure(Test, Tmap, directory, Tmap_type, noise_model_type)
    figure(figsize=(4,4))
    pd = (Test .- Tmap)./Tmap .* 100
    imshow( pd, cmap = "Greys")
    clb = colorbar()
    title("|% error| < $( round(maximum(abs.(pd)),digits=2) )",fontsize=15)
    clb.ax.tick_params(labelsize=15)
    axis("off")
    savefig("$(directory)/$(Tmap_type)_Tmap_percent_error_$(noise_model_type).png")
end

function plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmap, α, fftPSFs, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, directory, Tmap_type)
    plot_Tmap_figure(imgp, Tmap, directory, Tmap_type)
    image_Tmap_grid, noise_model_type = plot_image_figure(pp, imgp, Tmap, fftPSFs, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, directory, Tmap_type, "default")
    Test = plot_reconstruction_figure(pp, imgp, optp, recp, image_Tmap_grid, Tinit_flat, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, parallel, directory, Tmap_type, noise_model_type)
    plot_percent_error_figure(Test, Tmap, directory, Tmap_type, noise_model_type)
    
    noise_levels = ["1.0", "2.0", "5.0", "10.0"]
    for noise_level in noise_levels
        plot_Tmap_figure(imgp, Tmap, directory, Tmap_type)
        image_Tmap_grid, noise_model_type = plot_image_figure(pp, imgp, Tmap, fftPSFs, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, directory, Tmap_type, noise_level)
        Test = plot_reconstruction_figure(pp, imgp, optp, recp, image_Tmap_grid, Tinit_flat, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, parallel, directory, Tmap_type, noise_model_type)
        plot_percent_error_figure(Test, Tmap, directory, Tmap_type, noise_model_type)
    end
end

function load_MIT_Tmap(size, background_T, logo_T)
    object_loadfilename_MIT = "MIT$(size).csv"
    filename_MIT = @sprintf("ImagingOpt.jl/objdata/%s",object_loadfilename_MIT)
    diff = logo_T - background_T
    Tmap_MIT = readdlm(filename_MIT,',',Float64).* diff .+ background_T
end

function get_PSF_diffraction_limited(freq, pp, imgp, plan_nearfar)
    #incident = prepare_incident(pp,freq)
    incident = incident_field(pp.depth, freq, 1, pp.gridL, pp.cellL)
    n2f_kernel = prepare_n2f_kernel(pp,imgp,freq, plan_nearfar) 
    
    grid = range(-pp.gridL/2 + 0.5, pp.gridL/2 - 0.5, length = pp.gridL) .* pp.cellL
    phases = [-(2 * pi * freq ) * (sqrt(x^2 + y^2 + pp.F^2) - pp.F) for x in grid, y in grid]

    far = convolve(incident .* exp.(im .* phases ), n2f_kernel, plan_nearfar)
    PSF = far_to_PSF(far, imgp.objL + imgp.imgL, imgp.binL, pp.PSF_scaling, freq)
end

