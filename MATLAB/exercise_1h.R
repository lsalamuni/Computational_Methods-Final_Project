################################################################################
###                                                                          ###
###                              PACKAGES                                    ###
###                                                                          ###
################################################################################

pacotes <- c("dplyr", "readxl", "modelsummary", "tidyr", "fredr", "tibble",
             "lubridate", "ggplot2", "seasonal", "eurostat", "mFilter")

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){ 
  instalador <- pacotes[!pacotes %in% installed.packages()] 
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T)
} else {
  sapply(pacotes, require, character = T)
}



################################################################################
###                                                                          ###
###                              FRED API                                    ###
###                                                                          ###
################################################################################

fredr_set_key("...") #API key (insert key here)

key <- fredr_get_key()



################################################################################
###                                                                          ###
###                              RAW DATA                                    ###
###                                                                          ###
################################################################################

# 1.1. Real Gross Domestic Product for USA (GDPC1) - SAAR
gdp_us <- fredr(series_id = "GDPC1") %>%
  select(c(date, value)) %>%
  filter(date < "2025-01-01") %>%
  transmute(date, GDP_US = value/4) # SA level, billions 2017 USD (from SAAR)

# 1.2. Gross Domestic Product for Germany (CLVMNACSCAB1GQDE) - SA
gdp_de <- fredr(series_id = "CLVMNACSCAB1GQDE") %>%
  rename(GDP_DE = value) %>%
  filter(date < "2025-01-01") %>%
  select(c(date, GDP_DE)) # SA level, millions chained EUR (OECD)


# 2.1. Real Personal Consumption Expenditures for the USA (PCECC96) - SAAR
consumption_us <- fredr(series_id = "PCECC96") %>%
  select(c(date, value)) %>%
  filter(date < "2025-01-01") %>%
  transmute(date, CON_US = value/4) 

# 2.2. Private Final Consumption Expenditure in Germany (CPMNACSCAB1GQDE) - SA
consumption_de <- fredr(series_id = "CPMNACSCAB1GQDE") %>%
  select(c(date, value)) %>%
  filter(date < "2025-01-01") %>%
  rename(CON_DE = value) # millions, chain-linked EUR


# 3.1. Real Gross Private Domestic Investment in the US (GPDIC1) - SAAR
inv_us <- fredr(series_id = "GPDIC1") %>%
  filter(date < "2025-01-01") %>%
  transmute(date, INV_US = value/4) # SA quarterly level (billions, chained 2017 USD)

# 3.2. Gross Fixed Capital Formation in Germany (DEUGFCFQDSNAQ) - SA
inv_de <- fredr(series_id = "DEUGFCFQDSNAQ") %>%
  transmute(date, INV_DE = value) # millions, chained 2010 EUR


# 4.1. Nonfarm Business Sector: Hours Worked for All Workers in the US (HOANBS) - SA
hours_us <- fredr(series_id = "HOANBS") %>%
  filter(date < "2025-01-01") %>%
  dplyr::transmute(date, HOURS_US = value)  # Index 2017 = 100

# 4.2. Hours Worked for All Workers in Germany (I15_HW) - NSA --> need for adjustment (next session)
hrs_de_raw <- get_eurostat("namq_10_a10_e", time_format = "date")

hours_de <- hrs_de_raw %>%
  filter(geo == "DE" &
           unit == "I15_HW" &  # Index 2015 = 100
           na_item == "EMP_DC" & # Employment domestic concept
           s_adj == "NSA" &
           nace_r2 == "TOTAL") %>% # Total economy
  rename(date = TIME_PERIOD,
         HOURS_DE = values) %>%
  filter(date < "2025-01-01") %>%
  select(c(date, HOURS_DE))


# 5.1. Nonfarm Business Sector: Labor Productivity (Output per Hour) for All Workers (OPHNFB) - SA
prod_us <- fredr("OPHNFB") %>%
  filter(date < "2025-01-01") %>%
  transmute(date, PROD_US = value)

# 5.2. Unit Labor Costs: Early Estimate of Quarterly Unit Labor Costs (ULC) Indicators: Labor Productivity for Germany (ULQELP01DEQ661S) - SA
prod_de <- fredr("ULQELP01DEQ661S") %>%
  filter(date < "2025-01-01") %>%
  transmute(date, PROD_DE = value)


# 6.1. Business Tendency Surveys: Rate of Capacity Utilisation: Economic Activity: Manufacturing: Current for United States (BSCURT02USQ160S) - SA
util_us <- fredr("BSCURT02USQ160S") %>% 
  filter(date < "2025-01-01") %>%
  transmute(date, UTIL_US = value)

# 6.2. Business Tendency Surveys: Rate of Capacity Utilisation: Economic Activity: Manufacturing: Current for Germany (BSCURT02DEQ160S) - SA
util_de <- fredr("BSCURT02DEQ160S") %>%
  filter(date < "2025-01-01") %>%
  transmute(date, UTIL_DE = value)



################################################################################
###                                                                          ###
###                             HP DETRENDING                                ###
###                                                                          ###
################################################################################

# 1. Merge all German datasets to find common period

# 1.1. US
us_merged <- gdp_us %>%
  full_join(consumption_us, by = "date") %>%
  full_join(inv_us, by = "date") %>%
  full_join(hours_us, by = "date") %>%
  full_join(prod_us, by = "date") %>%
  full_join(util_us, by = "date") %>%
  arrange(date) %>%
  filter(complete.cases(.)) # Keep only rows with all variables

# 1.2. Germany
german_merged <- gdp_de %>%
  full_join(consumption_de, by = "date") %>%
  full_join(inv_de, by = "date") %>%
  full_join(hours_de, by = "date") %>%
  full_join(prod_de, by = "date") %>%
  full_join(util_de, by = "date") %>%
  arrange(date) %>%
  filter(complete.cases(.)) # Keep only rows with all variables


# 2. Seasonally adjust German hours using X-13ARIMA
hours_de_ts <- ts(german_merged$HOURS_DE,
                  start = c(year(min(german_merged$date)),
                            quarter(min(german_merged$date))),
                  frequency = 4)

hours_de_seas <- seas(hours_de_ts)

german_merged$HOURS_DE <- as.numeric(final(hours_de_seas))


# 3. Apply HP filter to each series individually (λ = 1600 for quarterly data)

# 3.1. US HP filtering
gdp_us_hp <- hpfilter(log(us_merged$GDP_US), freq = 1600)
con_us_hp <- hpfilter(log(us_merged$CON_US), freq = 1600)
inv_us_hp <- hpfilter(log(us_merged$INV_US), freq = 1600)
hours_us_hp <- hpfilter(log(us_merged$HOURS_US), freq = 1600)
prod_us_hp <- hpfilter(log(us_merged$PROD_US), freq = 1600)
util_us_hp <- hpfilter(log(us_merged$UTIL_US), freq = 1600)

# 3.1.1. US dataframe with HP-filtered data (cyclical components in %)
us_hp_data <- data.frame(date = us_merged$date,
                         GDP = gdp_us_hp$cycle * 100,
                         Consumption = con_us_hp$cycle * 100,
                         Investment = inv_us_hp$cycle * 100,
                         Hours = hours_us_hp$cycle * 100,
                         Productivity = prod_us_hp$cycle * 100,
                         Utilization = util_us_hp$cycle * 100)

# 3.2. German HP filtering  
gdp_de_hp <- hpfilter(log(german_merged$GDP_DE), freq = 1600)
con_de_hp <- hpfilter(log(german_merged$CON_DE), freq = 1600)
inv_de_hp <- hpfilter(log(german_merged$INV_DE), freq = 1600)
hours_de_hp <- hpfilter(log(german_merged$HOURS_DE), freq = 1600)
prod_de_hp <- hpfilter(log(german_merged$PROD_DE), freq = 1600)
util_de_hp <- hpfilter(log(german_merged$UTIL_DE), freq = 1600)

# 3.2.1. Create German dataframe with HP-filtered data (cyclical components in %)
de_hp_data <- data.frame(date = german_merged$date,
                         GDP = gdp_de_hp$cycle * 100,
                         Consumption = con_de_hp$cycle * 100,
                         Investment = inv_de_hp$cycle * 100,
                         Hours = hours_de_hp$cycle * 100,
                         Productivity = prod_de_hp$cycle * 100,
                         Utilization = util_de_hp$cycle * 100)


# 4. Get rid of unnecessary data
rm(con_de_hp, con_us_hp, consumption_de, consumption_us, gdp_de, gdp_de,gdp_de_hp,
   gdp_us, gdp_us_hp, german_merged, hours_de, hours_de_hp, hours_de_seas, hours_us, 
   hours_us_hp, hrs_de_raw, inv_de, inv_de_hp, inv_us, inv_us_hp, prod_de, prod_de_hp,
   prod_us, prod_us_hp, us_merged, util_de, util_us, util_de_hp, util_us_hp)


################################################################################
###                                                                          ###
###                         BUSINESS CYCLE STATISTICS                        ###
###                                                                          ###
################################################################################

# 1. US Statistics
us_stats <- data.frame(Variable = c("Output", "Consumption", "Investment", "Hours worked", "Productivity", "Utilization"),
                       Standard_deviation = c(sd(us_hp_data$GDP),
                                              sd(us_hp_data$Consumption),
                                              sd(us_hp_data$Investment),
                                              sd(us_hp_data$Hours),
                                              sd(us_hp_data$Productivity),
                                              sd(us_hp_data$Utilization)),
                       Correlation_with_output = c(cor(us_hp_data$GDP, us_hp_data$GDP),
                                                   cor(us_hp_data$GDP, us_hp_data$Consumption),
                                                   cor(us_hp_data$GDP, us_hp_data$Investment),
                                                   cor(us_hp_data$GDP, us_hp_data$Hours),
                                                   cor(us_hp_data$GDP, us_hp_data$Productivity),
                                                   cor(us_hp_data$GDP, us_hp_data$Utilization)),
                       Autocorrelation = c(cor(us_hp_data$GDP[-nrow(us_hp_data)], us_hp_data$GDP[-1]),
                                           cor(us_hp_data$Consumption[-nrow(us_hp_data)], us_hp_data$Consumption[-1]),
                                           cor(us_hp_data$Investment[-nrow(us_hp_data)], us_hp_data$Investment[-1]),
                                           cor(us_hp_data$Hours[-nrow(us_hp_data)], us_hp_data$Hours[-1]),
                                           cor(us_hp_data$Productivity[-nrow(us_hp_data)], us_hp_data$Productivity[-1]),
                                           cor(us_hp_data$Utilization[-nrow(us_hp_data)], us_hp_data$Utilization[-1])))


# 2. German Statistics
de_stats <- data.frame(Variable = c("Output", "Consumption", "Investment", "Hours worked", "Productivity", "Utilization"),
                       Standard_deviation = c(sd(de_hp_data$GDP),
                                              sd(de_hp_data$Consumption),
                                              sd(de_hp_data$Investment),
                                              sd(de_hp_data$Hours),
                                              sd(de_hp_data$Productivity),
                                              sd(de_hp_data$Utilization)),
                       Correlation_with_output = c(cor(de_hp_data$GDP, de_hp_data$GDP),
                                                   cor(de_hp_data$GDP, de_hp_data$Consumption),
                                                   cor(de_hp_data$GDP, de_hp_data$Investment),
                                                   cor(de_hp_data$GDP, de_hp_data$Hours),
                                                   cor(de_hp_data$GDP, de_hp_data$Productivity),
                                                   cor(de_hp_data$GDP, de_hp_data$Utilization)),
                       Autocorrelation = c(cor(de_hp_data$GDP[-nrow(de_hp_data)], de_hp_data$GDP[-1]),
                                           cor(de_hp_data$Consumption[-nrow(de_hp_data)], de_hp_data$Consumption[-1]),
                                           cor(de_hp_data$Investment[-nrow(de_hp_data)], de_hp_data$Investment[-1]),
                                           cor(de_hp_data$Hours[-nrow(de_hp_data)], de_hp_data$Hours[-1]),
                                           cor(de_hp_data$Productivity[-nrow(de_hp_data)], de_hp_data$Productivity[-1]),
                                           cor(de_hp_data$Utilization[-nrow(de_hp_data)], de_hp_data$Utilization[-1])))


# 3. Round to 3 decimal places for presentation
us_stats[, 2:4] <- round(us_stats[, 2:4], 3)
de_stats[, 2:4] <- round(de_stats[, 2:4], 3)

