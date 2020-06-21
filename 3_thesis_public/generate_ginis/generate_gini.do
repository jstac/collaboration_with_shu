


clear all

* log using ${data_dir}/generate_ginis.log, text replace 



global data_dir "/Users/shuhu/Desktop/1_papers_w:john/3_thesis/generate_ginis" 
global sav_dir "/Users/shuhu/Desktop/1_papers_w:john/3_thesis/generate_ginis" 


clear
set more off



/* Load SCF data */
use "${data_dir}/SCF_plus.dta", clear


/* Remove SCF components as described in Appendix B of Dettling et al. (2015) */
gen SCFnw = ffanw - life - ofin - vehi - onfin + othdebt
gen SCFfin = ffafin - life - ofin
gen SCFnfin = ffanfin - vehi - onfin
gen SCFhouse = house
gen SCFbus   = ffabus
gen SCFtdebt = tdebt - othdebt
gen SCFpdebt = pdebt - othdebt
gen SCFhdebt = hdebt

/* Construct wealth measures upstream rather than downstream */
replace SCFnfin = house + ffabus
replace SCFnw = SCFnfin + SCFfin - SCFtdebt

collapse SCF* [aw = wgtI95W95], by(yearmerge)

/* Construction from FA data
NW = FIN + NFIN - TDEBT

FIN   = DEPOS + BONDS + CORPEQUITY + MFUND + DCPEN
NFIN  = BUS + HOUSE
TDEBT = HDEBT + PDEBT

DEPOS       = LM153091003(A) + FL153020005(A) + FL153030005(A) + FL153034005(A)
BONDS       = FL153061105(A) + FL153061705(A) + FL153062005(A) + FL153063005(A)
CORPEQUITY  = LM153064105(A)
MFUND       = LM153064205(A)
DCPEN       = FL574090055(A) + FL224090055(A) + FL344090025(A)  

Pension in defined contribution plans
FL574090055(A) from table L.118
FL224090055(A) from table L.120
FL344090025(A) from table L.119c

BUS   = LM152090205(A)
HOUSE = LM155035015(A)

HDEBT = FL153165105(A)
PDEBT = FL153166000(A)
*/



/* Load SCF data */

qui use year* id impnum ffanw tinc *groups wgtI95W95 using "${data_dir}/SCF_plus.dta", clear
		
qui levelsof yearmerge, clean local(myears)

	foreach dvar in tinc ffanw { 

		foreach myear of local myears {
		
		di "`myear'"
		
		* compute percentiles
		qui _pctile `dvar' if yearmerge == `myear' [aw=wgtI95W95], p(99)

		local P99 = `r(r1)'

		qui replace `dvar'groups = 4 if yearmerge == `myear' & `dvar' > `P99' & `dvar' < .

		} // myear
	
	} // dvar

	foreach dvar in tinc ffanw { 
	
	qui gen gini_`dvar' = .
	qui gen gini_`dvar'_B99 = .
	qui gen gini_`dvar'_B90 = .

		foreach myear of local myears {
		
		di "`myear'"

		* Ginis
		qui sgini `dvar' [aw=wgtI95W95] if yearmerge == `myear'
		qui replace gini_`dvar' = r(coeff) if yearmerge == `myear'
		
		qui sgini `dvar' [aw=wgtI95W95] if yearmerge == `myear' & `dvar'groups < 4
		qui replace gini_`dvar'_B99 = r(coeff) if yearmerge == `myear'
		
		qui sgini `dvar' [aw=wgtI95W95] if yearmerge == `myear' & `dvar'groups < 3
		qui replace gini_`dvar'_B90 = r(coeff) if yearmerge == `myear'

		} // myear
	
	} // dvar
	
collapse gini*, by(yearmerge)

	foreach var of varlist gini* {
	replace `var' = round(`var' * 100)
	}
	
export excel _all using "${sav_dir}/ginis_all_b99_b90.xlsx", firstrow (var) replace

outsheet _all using "${sav_dir}/ginis_all_b99_b90.csv", comma nolabel

* log close _all
