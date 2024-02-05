function process_opt(presicion, parallel, opt_date, opt_id, pname)
    directory = "ImagingOpt.jl/optdata/$opt_id"
    
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
    
    ################################# plot objective and alpha values throughout opt #################################
    file_save_objective_vals = "$directory/objective_vals_$opt_date.csv"
    objdata = readdlm(file_save_objective_vals,',')
    figure(figsize=(22,10))
    suptitle(L"\mathrm{objective \  data } ,  \langle \frac{|| T - T_{est} ||^2}{  || T ||^2} \rangle_{T}")
    subplot(2,2,1)
    plot(objdata,".-")

    subplot(2,2,2)
    semilogy(objdata,".-")

    file_save_best_objective_vals = "$directory/best_objective_vals_$opt_date.csv"
    objdata_best = readdlm(file_save_best_objective_vals,',')
    subplot(2,2,3)
    plot(objdata_best,".-",color="orange")

    subplot(2,2,4)
    semilogy(objdata_best,".-",color="orange")
    tight_layout()
    savefig("$directory/objective_vals_$opt_date.png")
    
    if optp.optimize_alpha
        file_save_alpha_vals = "$directory/alpha_vals_$opt_date.csv"   
        alpha_vals = readdlm(file_save_alpha_vals,',')
        
        figure(figsize=(20,6))
        suptitle(L"\alpha \ \mathrm{values }")
        subplot(1,2,1)
        plot(alpha_vals,".-")

        subplot(1,2,2)
        semilogy(alpha_vals,".-")
        tight_layout()
        savefig("$directory/alpha_vals_$opt_date.png")
    end
    
    ################################# INITIAL geoms, alpha, and PSFs #################################
    #TO DO: if starting from random metasurface, save and then reload geoms here
    geoms_init = prepare_geoms(params)
    if parallel
        fftPSFs_init = ThreadsX.map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    else
        fftPSFs_init = map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    end
    α_init = optp.αinit
    
    #plot initial PSFs
    plot_PSFs(opt_date, directory, params, freqs, surrogates, plan_nearfar, geoms_init, parallel, "initial")
    
    #save initial random reconstruction
    plot_reconstruction(opt_date, directory, params, freqs, Tinit_flat, Tmaps_random, noises_random, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs_init, α_init, parallel, iqi, "initial", "random")
    
    #save initial MIT reconstruction
    plot_reconstruction(opt_date, directory, params, freqs, Tinit_flat, Tmaps_MIT, noises_random, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs_init, α_init, parallel, iqi, "initial", "MIT")
    
    #reconstructions for different noise levels, random and MIT Tmaps
    plot_reconstruction_fixed_noise_levels(opt_date, directory, params, freqs, Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_init, α_init, parallel, iqi, "initial", "random")
    
    plot_reconstruction_fixed_noise_levels(opt_date, directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_init, α_init, parallel, iqi, "initial", "MIT")
    
    #reconstruction with different alpha values, random and MIT Tmaps, 2% noise
    plot_reconstruction_different_alpha_vals(opt_date, directory, params, freqs, Tinit_flat, Tmaps_random[1], imgp.noise_level, plan_nearfar, plan_PSF, weights, fftPSFs_init, [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], parallel, iqi, "initial", "random")
    
    plot_reconstruction_different_alpha_vals(opt_date, directory, params, freqs, Tinit_flat, Tmap_MIT, imgp.noise_level, plan_nearfar, plan_PSF, weights, fftPSFs_init, [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], parallel, iqi, "initial", "MIT")
    
    ################################# OPTIMIZED geoms, alpha, and PSFs #################################
    geoms_filename = "$directory/geoms_optimized_$opt_date.csv"
    geoms_optimized = reshape(readdlm(geoms_filename,','),pp.gridL, pp.gridL )
    if parallel
        fftPSFs_optimized = ThreadsX.map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_optimized, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    else
        fftPSFs_optimized = map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_optimized, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    end
    if optp.optimize_alpha
        α_optimized = alpha_vals[end]
    else
        α_optimized = optp.αinit
    end
    
    #plot optimized PSFs
    plot_PSFs(opt_date, directory, params, freqs, surrogates, plan_nearfar, geoms_optimized, parallel, "optimized")
    
    #save optimized random reconstruction
    plot_reconstruction(opt_date, directory, params, freqs, Tinit_flat, Tmaps_random, noises_random, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random")
    
    #save optimized MIT reconstruction
    plot_reconstruction(opt_date, directory, params, freqs, Tinit_flat, Tmaps_MIT, noises_random, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT")
    
    #reconstructions for different noise levels, random and MIT Tmaps (optimized alpha and alpha=0)
    plot_reconstruction_fixed_noise_levels(opt_date, directory, params, freqs, Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random")
    
    plot_reconstruction_fixed_noise_levels(opt_date, directory, params, freqs, Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, 0, parallel, iqi, "optimized", "random")
    
    plot_reconstruction_fixed_noise_levels(opt_date, directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT")
    
    plot_reconstruction_fixed_noise_levels(opt_date, directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, 0, parallel, iqi, "optimized", "MIT")
    
    ################################# plot geoms init and optimized side by side #################################
    figure(figsize=(16,5))
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
    
    ################################# get transmissions for initial and optimized metasurface #################################
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
    
    dict_output_optimized = Dict("transmission_relative_to_no_lens" => transmission_relative_to_no_lens_optimized)
    #save output data in json file
    output_data_filename = "$(directory)/transmissions_optimized.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output_optimized)
    end
    
    ################################# figure plots #################################
    figure_plots_intial_directory = "ImagingOpt.jl/optdata/$opt_id/figure_plots_initial"
    if ! isdir(figure_plots_intial_directory)
        Base.Filesystem.mkdir( figure_plots_intial_directory )
    end
    plot_geoms_figure(geoms_init, figure_plots_intial_directory)
    plot_PSFs_figure(pp, imgp, freqs, surrogates, geoms_init, plan_nearfar, parallel, figure_plots_intial_directory)
    
    plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmap_MIT, α_init, fftPSFs_init, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, figure_plots_intial_directory, "MIT")
    plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmaps_random[1], α_init, fftPSFs_init, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, figure_plots_intial_directory, "random")
    
    
    figure_plots_optimized_directory = "ImagingOpt.jl/optdata/$opt_id/figure_plots_optimized"
    if ! isdir(figure_plots_optimized_directory)
        Base.Filesystem.mkdir( figure_plots_optimized_directory )
    end
    plot_geoms_figure(geoms_optimized, figure_plots_optimized_directory)
    plot_PSFs_figure(pp, imgp, freqs, surrogates, geoms_optimized, plan_nearfar, parallel, figure_plots_optimized_directory)
    
    plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmap_MIT, α_optimized, fftPSFs_optimized, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, figure_plots_optimized_directory, "MIT")
    plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmaps_random[1], α_optimized, fftPSFs_optimized, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, figure_plots_optimized_directory, "random")
    
    ################################# reconstructions on different temperature ranges #################################
    more_reconstructions_directory = "ImagingOpt.jl/optdata/$opt_id/more_reconstructions"
    if ! isdir(more_reconstructions_directory)
        Base.Filesystem.mkdir(more_reconstructions_directory)
    end
    
    bg = (imgp.lbT + imgp.ubT)/2
    logo = imgp.lbT + (imgp.ubT - imgp.lbT)*(1/4)
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = (imgp.lbT + imgp.ubT)/2
    logo = imgp.ubT 
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = (imgp.lbT + imgp.ubT)/2
    logo = imgp.lbT 
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = imgp.lbT 
    logo = imgp.ubT 
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = (imgp.lbT + imgp.ubT)/2 - 1
    logo = (imgp.lbT + imgp.ubT)/2 + 1
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 263.15
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 280
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 296
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 300
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 310
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 443.15
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 295
    logo = 623.15
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 263.15
    logo = 623.15
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    bg = 280
    logo = 310
    Tmap_MIT = load_MIT_Tmap(imgp.objL, bg, logo )
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT_$(bg)_$(logo)")
    
    lbT = 263.15
    ubT = 623.15
    Tmap_random = rand(lbT:eps():ubT,imgp.objL, imgp.objL)
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_random, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random_$(lbT)_$(ubT)")
    
    lbT = 280
    ubT = 310
    Tmap_random = rand(lbT:eps():ubT,imgp.objL, imgp.objL)
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_random, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random_$(lbT)_$(ubT)")
    
    lbT = 294
    ubT = 296
    Tmap_random = rand(lbT:eps():ubT,imgp.objL, imgp.objL)
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_random, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random_$(lbT)_$(ubT)")
    
    lbT = (imgp.lbT + imgp.ubT)/2 - 1
    ubT = (imgp.lbT + imgp.ubT)/2 + 1
    Tmap_random = rand(lbT:eps():ubT,imgp.objL, imgp.objL)
    plot_reconstruction_fixed_noise_levels(opt_date, more_reconstructions_directory, params, freqs, Tinit_flat, Tmap_random, [0.01; 0.02; 0.04; 0.05; 0.08; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random_$(lbT)_$(ubT)")
    
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
    tight_layout()
    savefig("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_$(@sprintf "%.4e" α )_fixed_noises_$(opt_date).png")
    
    open("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_$(@sprintf "%.4e" α )_fixed_noises_MSEs_$(opt_date).csv", "w") do io
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
    tight_layout()
    savefig("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_noise_$(noise_level)_different_alpha_vals_$(opt_date).png")
    
    open("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_noise_$(noise_level)_different_alpha_vals_MSEs_$(opt_date).csv", "w") do io
        writedlm(io, [alpha_vals, MSEs],',')
    end
end


function plot_PSFs(opt_date, directory, params, freqs, surrogates, plan_nearfar, geoms, parallel, geoms_type)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp
    
    if parallel
        PSFs = ThreadsX.map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
    else
        PSFs = map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
    end
    
    figure(figsize=(20,9))
    #assumes 21 PSFs
    for i = 1:21
        subplot(3,7,i)
        imshow(PSFs[i])
        colorbar()
        title("PSF for ν = $(round(freqs[i],digits=3) )")
    end
    tight_layout()
    savefig("$directory/PSFs_$(geoms_type)_$opt_date.png")
end

function get_transmission_relative_to_no_lens(pp, imgp, Tmap, fftPSFs_freespace, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
    B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
    image_Tmap_grid_freespace = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs_freespace, freqs, weights, plan_nearfar, plan_PSF, parallel)
    image_Tmap_grid_nonoise = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
    transmission = round(sum(image_Tmap_grid_nonoise) / sum(image_Tmap_grid_freespace) * 100,digits=2)
end

function plot_geoms_figure(geoms, directory)
    figure(figsize=(5,5))
    imshow(geoms, cmap="Greys_r")
    axis("off")
    savefig("$(directory)/geoms.png")
end

function plot_PSFs_figure(pp, imgp, freqs, surrogates, geoms, plan_nearfar, parallel, directory)
    #plot PSFs (make sure there are only 21 of them)
    if parallel
        PSFs = ThreadsX.map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
    else
        PSFs = map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
    end
    
    figure(figsize=(12,6))
    for i = 1:21
        subplot(3,7,i)
        imshow(PSFs[i])
        wavelength = pp.wavcen ./ freqs[i]
        title("λ = $(round(wavelength,digits=2) ) μm",fontsize=15)
        axis("off")
    end
    tight_layout()
    savefig("$(directory)/PSFs.png")
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
    #otherwise, noise_model_type is equal to noise_level 
        noise_model_type = "$(noise_model_type)_$(round(noise_level ,digits=4))"
    else
        noise_level = parse(Float64,noise_model_type) / 100
        noise = noise_level .* randn(imgp.imgL, imgp.imgL);
        image_Tmap_grid = make_image(pp, imgp, true, B_Tmap_grid, fftPSFs, freqs, weights, noise, 0, plan_nearfar, plan_PSF, parallel)
    end
    
    figure(figsize=(4,4))
    imshow(image_Tmap_grid)
    #colorbar()
    axis("off")
    savefig("$(directory)/$(Tmap_type)_image_$(noise_model_type).png")
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



