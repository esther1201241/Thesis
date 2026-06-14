/*********************************************
 * OPL 20.1.0.0 Model
 * Author: Esther
 * Base model: no emergency trucks
 * Promotion budget based on expected realized discount expenditure
 *********************************************/
string runName = ...;
// ---------- Dimensions ----------
int nI = ...;   // number of products
int nT = ...;   // number of periods
int nS = ...;   // number of scenarios
int nB = ...;
range B = 1..nB;



range I = 1..nI;
range T = 1..nT;
range S = 1..nS;

// ---------- Parameters ----------
float pi[S] = ...;                 // scenario probabilities
float dbarBase[I][T][B] = ...;       // baseline demand
float u[I] = ...;                  // own-promotion uplift
float gamma[I][I] = ...;           // cross-item spillover when another product is promoted
float jointGamma[I][I] = ...;      // additional interaction effect when both products are promoted
float g[I] = ...;                  // post-promotion dip
float Ftruck = ...;                // fixed cost for regular pre-booked trucks
int maxRegularTrucks[T] = ...;     
float a[I] = ...;                  
float Cap = ...;                   
float P[I] = ...;                  // regular selling price
float minExpectedFillRate = ...; 	// Minimum expected fill rate 
float ownPromoMult[S] = ...;   // uncertainty in own-promotion uplift
float compPromoMult[S] = ...;  // uncertainty in complementarity effects
int baseScenario[S] = ...;


//int fixedX[I][T] = ...;
//int fixedY[T] = ...;
int useComplementarity = ...; // 1 = include complementarity, 0 = exclude complementarity

// Cost of goods sold is not directly observed, so it is approximated as a
// fixed share of the selling price. With cogsRate = 0.70, the implied gross
// margin before discounts and operating costs is 30% of the selling price.
float cogsRate = ...;
dexpr float COGS[i in I] = cogsRate * P[i];
//dexpr float grossMargin[i in I] = P[i] - COGS[i];

// Holding costs are based on the value of inventory, proxied by COGS, and
// converted from an annual rate to the length of one model period.
float holdingCostAnnualRate = ...;
float weeksPerPeriod = ...;
dexpr float h[i in I] = holdingCostAnnualRate * COGS[i] * weeksPerPeriod / 52;

// Variable shipment/handling cost per shipped unit, expressed as a percentage
// of COGS rather than the full selling price.
float shipmentCostRate = ...;
dexpr float cship[i in I] = shipmentCostRate * COGS[i];

// Extra lost-sales penalty. The model already loses the gross margin when a
// sale is not fulfilled, so this parameter only captures additional service,
// goodwill, or customer-dissatisfaction costs.
float lostSalesPenaltyMultiplier = ...;
dexpr float pso[i in I] = lostSalesPenaltyMultiplier * P[i];
float delta[I] = ...;              // consumer-facing discount rate

// The observed discount depth is the consumer-facing markdown. The dataset does
// not identify whether the retailer or supplier funds the markdown. Therefore,
// only this share is treated as a cost borne by the retailer. For example,
// retailerPromoShare = 0.25 means the retailer bears 25% of the observed markdown.
float retailerPromoShare = ...;

float M[I][T] = ...;               // big-M for linearization
float I0[I] = ...;                 // initial inventory
int promo0[I] = ...;               // promotion status before period 1, usually 0
int maxPromosPerProduct[I] = ...;  // maximum number of promotions per product over the horizon
int minPromosPerProduct[I] = ...;  // optional minimum number of promotions per product; set to 0 to deactivate
int maxPromosPerPeriod[T] = ...;   // maximum number of products promoted in each period
int maxTotalPromos = ...;          // maximum amount of promotions over the entire timeline

//float FEmergencyTruck = ...;       // fixed cost for emergency trucks used after scenario is known
// ---------- Decision variables ----------
// First-stage decisions: made before the demand scenario is known.
dvar boolean x[I][T];               // 1 if product i is promoted in period t
dvar boolean bothPromoted[I][I][T]; // 1 if products i and j are both promoted in period t
dvar int+ y[T];                     // number of regular trucks reserved in period t

// Second-stage recourse decisions: scenario-dependent.
dvar float+ q[I][T][S];             // shipped quantity
dvar float+ Inv[I][T][S];           // end-of-period inventory
dvar float+ z[I][T][S];             // fulfilled demand / sales
dvar float+ L[I][T][S];             // lost sales
dvar float+ w[I][T][S];             // linearization variable = x * z
//dvar int+ e[T][S];                // emergency trucks used in period t under scenario s

// ---------- Demand expression ----------
// Demand depends on the scenario-dependent baseline and the first-stage promotion plan.
dexpr float OwnPromotionEffect[i in I][t in T] =
      u[i] * x[i][t];

dexpr float CrossPromotionEffect[i in I][t in T] =
      sum(j in I : j != i) gamma[i][j] * x[j][t]* useComplementarity;

dexpr float JointEffect[i in I][t in T] =
      sum(j in I : j != i) jointGamma[i][j] * bothPromoted[i][j][t]* useComplementarity;

dexpr float PromotionInducedEffect[i in I][t in T] =
      OwnPromotionEffect[i][t]
    + CrossPromotionEffect[i][t]
    + JointEffect[i][t];

dexpr float D[i in I][t in T][s in S] =
      dbarBase[i][t][baseScenario[s]]
    + ownPromoMult[s] * OwnPromotionEffect[i][t]
    + compPromoMult[s] * (CrossPromotionEffect[i][t] + JointEffect[i][t])
    - (t == 1 ? g[i] * promo0[i] : g[i] * x[i][t-1]);

// Capacity used in each period and scenario.
dexpr float UsedCapacity[t in T][s in S] =
      sum(i in I) a[i] * q[i][t][s];

// Expected retailer-funded markdown per period.
// The full consumer-facing discount is delta[i] * P[i], but only the share
// retailerPromoShare is treated as borne by the retailer.
dexpr float ExpectedRetailerMarkdownUse[t in T] =
      sum(i in I, s in S) pi[s] * retailerPromoShare * delta[i] * P[i] * w[i][t][s];

// Expected full consumer-facing discount value per period. 
dexpr float ExpectedConsumerDiscountUse[t in T] =
      sum(i in I, s in S) pi[s] * delta[i] * P[i] * w[i][t][s];

// Scenario-specific retailer-funded markdown use, only for reporting.
dexpr float ScenarioRetailerMarkdownUse[t in T][s in S] =
      sum(i in I) retailerPromoShare * delta[i] * P[i] * w[i][t][s];

// ---------- Objective ----------
// Maximize expected gross profit after COGS, discounts, holding costs,
// shipment costs, lost-sales penalties, and regular truck reservation costs.
maximize
    sum(s in S) pi[s] *(
        sum(i in I, t in T)
            (
              P[i] * z[i][t][s]                 // sales revenue
            - COGS[i] * z[i][t][s]              // cost of goods sold
            - retailerPromoShare * delta[i] * P[i] * w[i][t][s]      // retailer-funded markdown
            - h[i] * Inv[i][t][s]               // inventory holding cost
            - pso[i] * L[i][t][s]               // extra lost-sales penalty
            - cship[i] * q[i][t][s]             // variable shipment/handling cost
 
            )
            //- sum(t in T) FEmergencyTruck * e[t][s] // emergency truck costs
            )
  - sum(t in T) Ftruck * y[t];

// ---------- Constraints ----------
subject to {
  
 //forall(i in I, t in T)    					// This constraint can be used to fix the promotion schedule
 // x[i][t] == fixedX[i][t];					
//forall(t in T)								// This constraint can be used to fix the truck amounts and schedule
 //     y[t] == fixedY[t];
   
   sum(s in S) pi[s] * sum(i in I, t in T) z[i][t][s]		// Fill rate, the model is not allowed to compensate by only lost sales
  															// Not incoprorated in the base model, so set to 0
>= minExpectedFillRate *
   sum(s in S) pi[s] * sum(i in I, t in T) D[i][t][s];
   
   // Inventory balance, first period
   forall(i in I, s in S)
      Inv[i][1][s] == I0[i] + q[i][1][s] - z[i][1][s];

   // Inventory balance, later periods
   forall(i in I, t in 2..nT, s in S)
      Inv[i][t][s] == Inv[i][t-1][s] + q[i][t][s] - z[i][t][s];

   // Demand is either fulfilled or lost. Lost sales are the base-case recourse option
   // when regular capacity and available inventory are insufficient.
   forall(i in I, t in T, s in S)
      z[i][t][s] + L[i][t][s] == D[i][t][s];

   // Limit on regular pre-booked trucks.
   forall(t in T)
      y[t] <= maxRegularTrucks[t];

   // Truck capacity in the base model.
   forall(t in T, s in S)
   UsedCapacity[t][s] <= Cap * y[t];
  
   // forall(t in T, s in S)  								// Optional constraint for when emergency trucks are used
    //  UsedCapacity[t][s] <= Cap * (y[t] + e[t][s]);

   // Linearization: w = x * z
   forall(i in I, t in T, s in S) {
      w[i][t][s] <= z[i][t][s];
      w[i][t][s] <= M[i][t] * x[i][t];
      w[i][t][s] >= z[i][t][s] - M[i][t] * (1 - x[i][t]);
   }

   // Diagonal is not used for joint effects.
   forall(i in I, t in T)
      bothPromoted[i][i][t] == 0;

   // Linearization: bothPromoted[i][j][t] = x[i][t] * x[j][t]
   forall(i in I, j in I : i != j, t in T) {
      bothPromoted[i][j][t] <= x[i][t];
      bothPromoted[i][j][t] <= x[j][t];
      bothPromoted[i][j][t] >= x[i][t] + x[j][t] - 1;
   }

   // Promotion-frequency constraints.
   // They represent limited promotional planning capacity, flyer/display space,
   // category-management restrictions, and promotion-calendar limitations.

   // Maximum number of promotions per product over the entire horizon.
   forall(i in I)
      sum(t in T) x[i][t] <= maxPromosPerProduct[i];

   // Optional minimum number of promotions per product. Set minPromosPerProduct[i] = 0, not active
   forall(i in I)
      sum(t in T) x[i][t] >= minPromosPerProduct[i];

   // Maximum number of products that may be promoted in the same period.
   forall(t in T)
      sum(i in I) x[i][t] <= maxPromosPerPeriod[t];
      
   // Promotion-slot budget over the entire planning horizon.
	// This limits the total number of product-period promotions selected.
	sum(i in I, t in T) x[i][t] <= maxTotalPromos;
}

// ---------- Output files ----------
execute ExportResults {
   var expectedDemand = 0;
   var expectedFulfilled = 0;
   var expectedLost = 0;
   var expectedShipped = 0;
   var expectedUsedCapacity = 0;
   var expectedAvailableCapacity = 0;
   var expectedRetailerMarkdownUse = 0;
   var expectedConsumerDiscountUse = 0;
   var maxScenarioPeriodRetailerMarkdownUse = 0;
   var totalPromotions = 0;
   var totalRegularTrucks = 0;

   for (var i in I) {
      for (var t in T) {
         totalPromotions += x[i][t];
      }
   }
   for (var t in T) {
      totalRegularTrucks += y[t];
      expectedAvailableCapacity += Cap * y[t];
      expectedRetailerMarkdownUse += ExpectedRetailerMarkdownUse[t];
      expectedConsumerDiscountUse += ExpectedConsumerDiscountUse[t];
      for (var s in S) {
         if (ScenarioRetailerMarkdownUse[t][s] > maxScenarioPeriodRetailerMarkdownUse) {
            maxScenarioPeriodRetailerMarkdownUse = ScenarioRetailerMarkdownUse[t][s];
         }
      }
   }
   for (var s in S) {
      for (var t in T) {
         expectedUsedCapacity += pi[s] * UsedCapacity[t][s];
      }
      for (var i in I) {
         for (var t in T) {
            expectedDemand += pi[s] * D[i][t][s];
            expectedFulfilled += pi[s] * z[i][t][s];
            expectedLost += pi[s] * L[i][t][s];
            expectedShipped += pi[s] * q[i][t][s];
         }
      }
   }

   var summary = new IloOplOutputFile(runName + "solution_summary.csv");
   summary.writeln("measure,value");
   summary.writeln("model,base_no_emergency_promo_frequency_with_cogs");
   summary.writeln("objective," + cplex.getObjValue());
   summary.writeln("total_promotions," + totalPromotions);
   summary.writeln("total_regular_trucks," + totalRegularTrucks);
   summary.writeln("expected_demand," + expectedDemand);
   summary.writeln("expected_fulfilled_sales," + expectedFulfilled);
   summary.writeln("expected_lost_sales," + expectedLost);
   summary.writeln("expected_fill_rate," + (expectedDemand > 0 ? expectedFulfilled / expectedDemand : 0));
   summary.writeln("expected_shipped_quantity," + expectedShipped);
   summary.writeln("expected_used_capacity," + expectedUsedCapacity);
   summary.writeln("expected_available_capacity," + expectedAvailableCapacity);
   summary.writeln("capacity_utilization," + (expectedAvailableCapacity > 0 ? expectedUsedCapacity / expectedAvailableCapacity : 0));
   summary.writeln("retailer_promo_share," + retailerPromoShare);
   summary.writeln("expected_retailer_funded_markdown," + expectedRetailerMarkdownUse);
   summary.writeln("expected_full_consumer_discount_value," + expectedConsumerDiscountUse);
   summary.writeln("max_scenario_period_retailer_funded_markdown," + maxScenarioPeriodRetailerMarkdownUse);
   summary.writeln("mip_relative_gap," + cplex.getMIPRelativeGap());
   summary.writeln("branch_and_cut_nodes," + cplex.getNnodes());
   summary.close();

   var promo = new IloOplOutputFile(runName + "promotion_plan.csv");
   promo.writeln("product,period,promoted");
   for (var i in I) {
      for (var t in T) {
         promo.writeln(i + "," + t + "," + x[i][t]);
      }
   }
   promo.close();

   var promoUsage = new IloOplOutputFile(runName + "promotion_activity_usage.csv");
   promoUsage.writeln("period,promotions_selected,max_promotions_allowed,expected_retailer_funded_markdown,expected_full_consumer_discount_value,max_realized_retailer_funded_markdown");
   for (var t in T) {
      var promosInPeriod = 0;
      var maxRealizedRetailerMarkdown = 0;
      for (var i in I) {
         promosInPeriod += x[i][t];
      }
      for (var s in S) {
         if (ScenarioRetailerMarkdownUse[t][s] > maxRealizedRetailerMarkdown) {
            maxRealizedRetailerMarkdown = ScenarioRetailerMarkdownUse[t][s];
         }
      }
      promoUsage.writeln(t + "," + promosInPeriod + "," + maxPromosPerPeriod[t] + "," + ExpectedRetailerMarkdownUse[t] + "," + ExpectedConsumerDiscountUse[t] + "," + maxRealizedRetailerMarkdown);
   }
   promoUsage.close();

   var trucks = new IloOplOutputFile(runName + "truck_plan.csv");
   trucks.writeln("period,regular_trucks_reserved");
   for (var t in T) {
      trucks.writeln(t + "," + y[t]);
   }
   trucks.close();

   var scen = new IloOplOutputFile(runName + "scenario_summary.csv");
   scen.writeln("scenario,probability,total_demand,fulfilled_sales,lost_sales,fill_rate,shipped_quantity,used_capacity,available_capacity");
   for (var s in S) {
      var scenDemand = 0;
      var scenFulfilled = 0;
      var scenLost = 0;
      var scenShipped = 0;
      var scenUsedCapacity = 0;
      var scenAvailableCapacity = 0;
      for (var t in T) {
         scenUsedCapacity += UsedCapacity[t][s];
         scenAvailableCapacity += Cap * y[t];
      }
      for (var i in I) {
         for (var t in T) {
            scenDemand += D[i][t][s];
            scenFulfilled += z[i][t][s];
            scenLost += L[i][t][s];
            scenShipped += q[i][t][s];
         }
      }
      scen.writeln(s + "," + pi[s] + "," + scenDemand + "," + scenFulfilled + "," + scenLost + "," + (scenDemand > 0 ? scenFulfilled / scenDemand : 0) + "," + scenShipped + "," + scenUsedCapacity + "," + scenAvailableCapacity);
   }
   scen.close();
}

execute {
  writeln("----- CPLEX SOLVER INFORMATION -----");
  writeln("Model: base model without emergency trucks, promotion-frequency limits, retailer-funded markdown share, with COGS");
  writeln("Objective value: ", cplex.getObjValue());
  writeln("CPLEX status: ", cplex.getCplexStatus());
  writeln("Number of variables: ", cplex.getNcols());
  writeln("Number of constraints: ", cplex.getNrows());
  writeln("Number of binary variables: ", cplex.getNbinVars());
  writeln("Number of integer variables: ", cplex.getNintVars());
  writeln("Number of non-zero coefficients: ", cplex.getNNZs());
  writeln("MIP relative gap: ", cplex.getMIPRelativeGap());
  writeln("Branch-and-cut nodes: ", cplex.getNnodes());
}
