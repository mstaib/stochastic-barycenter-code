#include <cxxopts.hpp>
#include <sys/stat.h>

#include "parse_args.hpp"

std::shared_ptr<options_struct> parse_args(int argc, char* argv[]) {
    
    std::shared_ptr<options_struct> opt_struct = std::make_shared<options_struct>();
    opt_struct->iters = 10000;
    opt_struct->experiment = "gaussian";
    opt_struct->subsets = 40;
    opt_struct->skip = 100;
    opt_struct->support = 1000;
    opt_struct->outdir = "";
    opt_struct->save_increment = 10000;
    opt_struct->stepsize = 0.5;
    opt_struct->moving_window_width = 0;
    opt_struct->drift_rate = 0;
    opt_struct->burnin_iters = 1000;
    opt_struct->full_sampler = false;
    opt_struct->num_datapoints = 0;

    try {
        cxxopts::Options options(argv[0], "Stochastic Wasserstein Barycenters");

        options.add_options()
            ("i,iters", "Number of iterations to run each thread", cxxopts::value<int>(), "N")
            ("e,experiment", "Which experiment to run", cxxopts::value<std::string>(), "EXPERIMENT")
            ("s,subsets", "Number of subsets to split into (for WASP)", cxxopts::value<int>(), "K")
            ("k,skip", "Number of timesteps between MCMC samples", cxxopts::value<int>(), "K")
            ("N,support", "Number of support points", cxxopts::value<int>(), "N")
            ("o,outdir", "Output directory for .h5 files", cxxopts::value<std::string>(), "OUTDIR")
            ("d,saveincrement", "How often to save .h5 files", cxxopts::value<int>(), "SAVEINC")
            ("a,stepsize", "Stepsize for gradient ascent", cxxopts::value<double>(), "STEPSIZE")
            ("w,movingwindow", "Width of histogram moving window", cxxopts::value<int>(), "WINDOWWIDTH")
            ("m,driftrate", "Rate of drift of VMF distributions", cxxopts::value<double>(), "DRIFTRATE")
            ("b,burniniters", "Number of burn-in iters for MCMC chain", cxxopts::value<int>(), "ITERS")
            ("f,fullsampler", "Whether to get samples from the full MCMC chain", cxxopts::value<bool>(), "FULLSAMPLER")
            ("p,datapoints", "Number of datapoint to use (for the skin example)", cxxopts::value<int>(), "DATAPOINTS")
        ;

        options.parse(argc, argv);

        if (options.count("iters")) {
            opt_struct->iters = options["iters"].as<int>();
        }

        if (options.count("experiment")) {
            opt_struct->experiment = options["experiment"].as<std::string>();
        }

        if (options.count("subsets")) {
            opt_struct->subsets = options["subsets"].as<int>();
        }

        if (options.count("skip")) {
            opt_struct->skip = options["skip"].as<int>();
        }

        if (options.count("support")) {
            opt_struct->support = options["support"].as<int>();
        }
        
        if (options.count("saveincrement")) {
            opt_struct->save_increment = options["saveincrement"].as<int>();
        }

        if (options.count("stepsize")) {
            opt_struct->stepsize = options["stepsize"].as<double>();
        }

        if (options.count("movingwindow")) {
            opt_struct->moving_window_width = options["movingwindow"].as<int>();
        }

        if (options.count("driftrate")) {
            opt_struct->drift_rate = options["driftrate"].as<double>();
        }

        if (options.count("burniniters")) {
            opt_struct->burnin_iters = options["burniniters"].as<int>();
        }
        
        if (options.count("fullsampler")) {
            opt_struct->full_sampler = options["fullsampler"].as<bool>();
        }

        if (options.count("datapoints")) {
            opt_struct->num_datapoints = options["datapoints"].as<int>();
        }

        if (options.count("outdir")) {
            std::string outdir = options["outdir"].as<std::string>();
            if (outdir.back() != '/') {
                outdir += '/';
            }
            
            struct stat info;
            if (stat(outdir.c_str(), &info) != 0) {
                std::cout << "Cannot access directory: " << outdir << std::endl;
                exit(1);
            }

            opt_struct->outdir = outdir;
        }

    } catch (const cxxopts::OptionException& e)
    {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }

    return opt_struct;
}