library(tidyverse)
library(ARTool)
library(lme4)
library(lmerTest)
library(ggplot2)
library(emmeans)
library(dplyr)
library(lme4)
library(stargazer)
library(car)
library(emmeans)
library(tidyr)
library(psych)  # for describeBy
library(ggbeeswarm)


do_stats <- function(csvfile, variable, ymin, ymax) {
    d <- read.csv(csvfile)
    cat(sprintf("\n\n\n*********************************************\n"))
    print(sprintf("Var to test: %s from %s", variable, csvfile))

    # trim the dataset
    # select cell type:
    d <- d %>% filter(cell_type == "pyramidal")
    # pick the relevant variables to keep track of (factors, subject id)
    testvars <- c(variable, "age_category", "sex", "cell_id")
    d2 <- d[, testvars]
    d2 <- drop_na(d2, variable) # remove empty rows.
    d2$age_category <- factor(d2$age_category) # make sure these are factors
    d2$sex <- factor(d2$sex)
    d2variable <- as.numeric(d2$variable) # make sure this is a number
    d2$cell_id <- as.character(d2$cell_id) # make sure this is a character
    # ad a subject column (subject = day of experiment)
    d2[c("Subject", "x", "slicecell")] <- str_split_fixed(d2$cell_id, "_", 3)
    keeps <- c(variable, "age_category", "sex", "cell_id", "Subject")
    d2 <- d2[, keeps]
    d2$Subject <- factor(d2$Subject)
    # d2[variable] <- gsub("[", "", d2[variable])
    # d2[variable] <- gsub("]", "", d2[variable])
    # reorder the levels of age_category for the plot
    d2$age_category  <- factor(d2$age_category,levels = c("Preweaning", "Pubescent", "Young Adult", "Mature Adult", "Old Adult", "unknown"))
    # print(head(d2, n = 190))
    # generate a qqplot and write to disk
    datestr <- format(Sys.time(), "%Y.%m.%d %X") # data of analysis/plotting
    qqfile <- sprintf("%s_age_qq.pdf", variable) # name of the qq plot file

    qqp <- ggplot(d2, aes_string(sample = variable, color = "age_category"), ) +
        geom_qq_band(bandType = "pointwise", mapping = aes(fill = "Normal", color = "age_category"), alpha = 0.5) +
        stat_qq_point() +
        stat_qq_line(linewidth = 0.5) +
        scale_fill_discrete(name = "Distribution") +
        facet_wrap(~sex) +
        labs(title = sprintf("%s by age", variable)) +
        theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
        labs(tag = sprintf("%s %s", qqfile, datestr)) +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
        theme(plot.tag.position = c(0.95, 0.01)) +
        theme(plot.tag = element_text(size = 6, hjust = 1))

    ggsave(file = qqfile, plot = qqp, width = 6, height = 4)
    cat(sprintf("saved qq plot: %s\n", qqfile))
    graphics.off()
    options(width=120)
    sa = describeBy(d2[, variable], d2$age_category)
    print(sa)
    cat(" \n")
    cat("*****ANOVA*****\n")
    # straight 1-way ANOVA (ignore sex)
    m_anova <- aov(d2[, variable] ~ age_category, data = d2)
    print(summary(m_anova))
    cat("\n")
    # run a linear mixed effects model
    # m_AR <- lmer(d2[, variable] ~ age_category * sex + (1 | Subject),  data = d2)
    # print(summary(m_AR))
    # a_AR <- anova(m_AR)
    # print(a_AR)

    # For Aligned Ranks Transform (ART) analysis - non parametric
    # analysis similar to ANOVA, can do 2-way, etc.
    # m <- art(AdaptRatio ~ age_category*sex + (1|animal_identifier), data=d2)
    # print(summary(m))
    # a = anova(m)
    # print(a)
    sum_test = unlist(summary(m_anova))
    print(names(sum_test))
    pvalue <- sum_test["Pr(>F)1"]
    print(pvalue)
    # run a post-hoc test
    if (pvalue < 0.05) {
        title <- sprintf("Posthoc pairs for %s\n", variable)
        cat(title)
        emm <- emmeans(m_anova, ~ age_category)
        tau_pairs <- pairs(emm, simple = "age_category") # interested in effect ofagee
        print(summary(tau_pairs))
        }
        else {
            cat("No significant differences found, no post-test\n")   
        }
    # generate a boxplot and write to disk
    plotfile <- sprintf("%s_Age.pdf", variable)
    cat("====================\n")
    gp <- ggplot(d2, aes_string(x = "age_category", y = variable, color="age_category")) +
        geom_boxplot(alpha = 1, notch=TRUE) +
        theme_bw() +
        xlab("Age") +
        ylab(variable) +
        # geom_point() +
        geom_beeswarm(alpha=0.5, cex=1, size=1, dodge.width=1.5, outlier.size=1.0, outlier.stroke=0.5) +
        scale_fill_manual(values = rep(NA, 2)) +
        scale_color_manual(values = c("pink", "light green", "sky blue", "brown", "grey", "black")) +
        ylim(ymin, ymax) +
        labs(title = sprintf("%s by Age", variable)) +
        theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
        labs(tag = sprintf("%s  %s", plotfile, datestr)) +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
        theme(plot.tag.position = c(1, 0.01)) +
        theme(plot.tag = element_text(size = 6, hjust = 1))

    ggsave(file = plotfile, plot = gp, width = 6, height = 4)

    return # all done.
}


# specify the csv files, analyze the variables, and plot with y scale settings
fn <- "spike_shapes.csv"
# # do the same analysis on different measures
do_stats(fn, "dvdt_rising_bestRs", 0, 1000)
do_stats(fn, "dvdt_falling_bestRs", 0, 1000)
do_stats(fn, "AP_HW_bestRs", 0, 1000)
do_stats(fn, "AP_thr_V_bestRs", -60, -20)
do_stats(fn, "AHP_depth_V_bestRs", -20, 0)

fn <- "rmtau.csv"
do_stats(fn, "Rin_bestRs", 0, 250)
do_stats(fn, "taum_bestRs", 0, 50)
do_stats(fn, "RMP_bestRs", -100, -40)

fn <- "firing_parameters.csv"
do_stats(fn, "AdaptRatio_bestRs", 0, 8)
do_stats(fn, "maxHillSlope_bestRs", 0, 1000)
do_stats(fn, "I_maxHillSlope_bestRs", 0, 1)
do_stats(fn, "FIMax_1_bestRs", 0, 1000)

# files <- c("spike_shapes.csv", "rmtau.csv", "firing_parameters.csv")
# variables <- list(
#     list(
#         list("dvdt_rising", 0, 750), list("dvdt_falling", 0, 200), list("AP_HW", 0, 2000),
#                  list("AP_thr_V", -60, -20), list("AHP_depth_V", -20, 0)
#                  ),
#     list(
#         list("Rin", 0, 600), list("taum", 0, 100), list("RMP", -100, -50)
#         ),
#     list(
#         list("AdaptRatio", 0, 8), list("maxHillSlope", 0, 500), list("I_maxHillSlope", 0, 0.8), list("FIMax_1", 0, 400)
#         )
# )


# for (i in 1:length(files)) {
#     fn <- files[i]
#     print(sprintf("File: %s", fn))

#     vars <- variables[i]
#     for (j in 1:length(vars)) {
#         pars <- vars[j]
#         print(sprintf("i: %d j: %d ", i, j))
#         print(pars)
#         # print(typeof(pars[1]))
#         # print(sprintf("      %c %d %d", pars[1], pars[2], pars[3]))
#         # do_stats(fn, pars[1], pars[2], pars[3])
#     }
# }
