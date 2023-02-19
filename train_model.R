#!/usr/bin/env Rscript

# Training Stan model with command line arguments; saving/plotting results automatically

# Reading command line arguments
args = commandArgs(trailingOnly=TRUE)

stan_file = args[1]
N_ITER = as.numeric(args[2])

expname = args[3]
proj_dir = './' # modify to select project folder for storing results

print(args)

# Load libraries / Setup
library(readxl)
library(readr)

library(rstan)
library(loo)
library(MCMCvis)
library(dplyr)
library(ggmcmc)
library(ggpubr)
library(grid)

openblasctl::openblas_set_num_threads(1) # avoid multi-threading within chain; might cause conflict
options(mc.cores = parallelly::availableCores())  

################################################################################
##############                   Train STAN MODEL                 ##############
################################################################################
# prep data
model_name = tools::file_path_sans_ext(basename(stan_file))
model_filename = paste0("Stan_models/", stan_file, ".stan")

cat("\n################# ", model_filename, " #################\n")
cat(read_file(model_filename))
cat("\n######################################################\n")

### Omitting data preparation steps
### Currently, the clinical data used in our model are not available for public access.

model_data = list(Nsubject=Nsubject, Ntime=Nweek, Nobs=nrow(df), 
                  t_obs= df$week, id_obs = df$id, MAL_obs=df$MAL_obs, 
                  MAL_init = MAL_init, train_dose = train_dose, 
                  train_time = train_time, MAL_pred_slope = 0.2)

model = stan(file = model_filename,  # Stan program
             data = model_data,      # named list of data
             chains = getOption("mc.cores", 1L),   # number of Markov chains
             iter = N_ITER,            # total number of iterations per chain
             control = list(max_treedepth = 15, adapt_delta = 0.9),
             seed = 1234,
             init = 'random'
             )

################################################################################
##############                   SAVE RESULTS                     ##############
################################################################################
# store model 
model_save_name = paste0(proj_dir, "Stanfits/", model_name, "_", expname,".rds")
saveRDS(model, model_save_name)

## Store LOO/WAIC result
log_lik = extract_log_lik(model, merge_chains = FALSE)
r_eff = relative_eff(exp(log_lik))
loo_out = loo(log_lik, r_eff = r_eff)
print(loo_out)

saveRDS(loo_out, paste0(proj_dir, "LOOs/LOO_", model_name, "_", expname,".rds"))
saveRDS(waic(log_lik), paste0(proj_dir, "LOOs/WAIC_", model_name, "_", expname,".rds"))

## Store MAL_fitted and CIs
pars = paste0('MAL[', df$week, ',', df$id, ']')
MAL_summary = summary(model, pars=pars, probs = c(0.025, 0.5, 0.975))$summary
df$MAL_fitted = MAL_summary[, "50%"]
df$MALp2.5 = MAL_summary[, "2.5%"]
df$MALp97.5 = MAL_summary[, "97.5%"]

result_dir = paste0('./model_results/', model_name, '_', expname, '/')
dir.create(result_dir, showWarnings=FALSE)

write.csv(df, paste0(result_dir, 'MAL_fits.csv'), row.names=FALSE)
print(Metrics::rmse(df$MAL_fitted, df$MAL_obs) )

################################################################################
##############                    PLOTS                           ##############
################################################################################
modelfitmcmc = As.mcmc.list(model)

intvl1 = c(0.025, 0.975)
intvl2 = c(0.005, 0.995)
intvl1_width = 0.15
intvl2_width = 0.5

font_settings = theme_classic() + 
  theme(axis.text.x = element_text(size = 13),
        axis.text.y = element_text(size = 20),
        axis.title = element_blank(),
        plot.title = element_text(color="black", size=16, hjust = 0))

## Predicted MALs (Figure 1.A)
p1 = ggplot(data=df) +
  # here add training weeks bars
  geom_ribbon(aes(x=week, ymin = MALp2.5, ymax=MALp97.5), fill = 'blue', alpha = 0.2) +
  geom_line(aes(x=week, y=MAL_fitted, color=factor(dose)), size = 1.5, alpha = 0.6) +
  geom_point(aes(x=week, y=MAL_obs, color=factor(dose)), size = 1.2) + 
  facet_wrap(~ id, ncol = 10, scales='fixed') +
  guides(color=guide_legend("dose")) +
  theme_classic() +
  theme(text = element_text(size = 14),
        strip.text = element_blank(),
        legend.position="right")

ps = list()
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
colors = function(subj_id){
  cols = gg_color_hue(4)
  return(cols[ceiling(subj_id/10)])
}

df_plot = df
df_plot$week = df_plot$week
drect=data.frame(x1=c(3,8,13), x2=c(4,9,14), y1=c(0,0,0), y2=c(5.5, 5.5, 5.5), training=c('1','2','3'), r=c(1,2,3))

for (i in 1:Nsubject) {
  tmp = subset(df_plot, id==i)
  color = colors(i)
  ps[[i]] = ggplot(data=tmp) +
  # here add training weeks bars
  geom_rect(data=drect, aes(xmin=x1, xmax=x2, ymin=y1, ymax=y2, group=r), 
            fill='azure3', alpha=c(0.3, 0.6, 0.9), show.legend = FALSE) +
  geom_ribbon(aes(x=week, ymin=MALp2.5, ymax=MALp97.5), fill='blue', alpha=0.2) +
  geom_line(aes(x=week, y=MAL_fitted), size=1.5, alpha=0.6, color=color) +
  geom_point(aes(x=week, y=MAL_obs), size=1.2, color=color) +
  theme_classic() + 
  scale_y_continuous(breaks=seq(0, 5, 1), name="MAL") +
  theme(axis.title.x = element_blank(),
          axis.title.y = element_blank()) +
  {if(i != "31")
      theme(axis.text.x = element_blank(),
            axis.text.y = element_blank()) } +
  scale_x_continuous(breaks= weeks_obs, name = "Weeks") +
  coord_cartesian(ylim=c(0, 5.5))
}

p1_new = ggarrange(plotlist = ps, nrow=4, ncol = 10, align='hv', common.legend = TRUE, legend='right', legend.grob=get_legend(p1))

fig_1a = annotate_figure(p1_new, left = textGrob("MAL", rot = 90, vjust = 1, gp = gpar(cex = 1.7)),
                    bottom = textGrob("Time (weeks)", gp = gpar(cex = 1.7)),
                    top = " ",
                    fig.lab = "A: Model fit to the MAL data, arranged by weekly dose of training",
                    fig.lab.pos='top.left', 
                    fig.lab.size=20
                    )

pdf(paste0(result_dir, "indiv_fits.pdf"), width = 13, height = 7)
print(fig_1a)
dev.off()

## Individual Parameters (Figure 1.B)
ps = list()
colors = list(
  "alpha" = "red",
  "beta" ="blue",
  "gamma" = "green"
)
titles = list(
  "alpha" = "Retention rate",
  "beta" ="Learning rate",
  "gamma" = "Self-training rate"
)

for (par in c('alpha', 'beta', 'gamma')){
  subs = 1:Nsubject  # c(2, 11, 25, 33)
  labs = parse(text = paste0(par, "[", subs, "]"))
  ci_par = ci(ggs(modelfitmcmc, family = paste0("^", par)), thick_ci = intvl1, thin_ci =intvl2)
  ci_par = ci_par[subs, ]
  ps[[par]] = ggplot(ci_par, aes(x=Parameter, ymin=low, ymax=high, lower = Low, middle=median, upper=High))+
    geom_boxplot(stat='identity',  width = intvl1_width, lwd=intvl2_width,  fill = colors[[par]], color = colors[[par]]) +
    guides(fill='none') +
    coord_flip() +
    scale_x_discrete(labels=labs) +
    ggtitle(titles[[par]]) +
    font_settings
}

figure1b =  ggarrange(plotlist=ps, ncol = 3, nrow = 1, widths=c(1.1,1,1))

figure1b = annotate_figure(figure1b, left = textGrob("Parameters", rot = 90, vjust = 1, gp = gpar(cex=1.7)),
                           bottom = textGrob("Posterior Credible Intervals", gp = gpar(cex=1.7)))

pdf(paste0(result_dir, "indiv_params.pdf"), width = 7.2, height = 1.1*Nsubject)
print(figure1b)
dev.off()

## Population parameters (Figure 1.C)
ci.pop <- ci(ggs(modelfitmcmc, family = "mu_alpha|mu_beta|mu_gamma"), thick_ci = intvl1, thin_ci = intvl2)
ci.pop$Parameter = ordered( ci.pop$Parameter, levels = c("mu_alpha", "mu_beta", "mu_gamma"))
font_settings = font_settings + theme(axis.text.x = element_text(size = 18))

pop <- ggplot(ci.pop, aes(x=Parameter, ymin=low, ymax=high, lower = Low, middle=median, upper=High))+
  geom_boxplot(data=ci.pop, stat='identity',  width = intvl1_width, lwd=intvl2_width,  
               fill = c("red", "blue", "green"), color = c("red", "blue","green")) +
  guides(fill='none')+
  coord_flip() +
  scale_x_discrete(labels= c(expression(mu[alpha]), expression(mu[beta]), expression(mu[gamma]))) +
  labs(y = "Posterior Credible Intervals") +
  labs(x = "Hyper-Parameters") +
  font_settings
figure1c = annotate_figure(pop, left = textGrob("Parameters", rot = 90, vjust = 1, gp = gpar(cex=1.7)),
                           bottom = textGrob("Posterior Credible Intervals", gp = gpar(cex=1.7)))

pdf(paste0(result_dir, "popul_params.pdf"), width = 7, height = 4)
print(figure1c)
dev.off()













