var jStat = require('jStat').jStat;
var mathjs = require('mathjs');

(function(exports) {

  'use strict'

  ////////////////////////////////////////////////////////////////////////////
  // Arm - one of the multi-armed bandit's arms, tracking observed rewards. //
  ////////////////////////////////////////////////////////////////////////////

  /**
   * @param {object} [args]
   * @param {int} [args.count] - number of attempts on this arm
   * @param {number} [args.sum] - total value accumulated on this arm
   * @constructor
     */
  function Arm(args) {
    this.count = args && args.count || 0
    this.sum = args && args.sum || 0
  }

  /**
   * @param {number} value
   *
   * Increments number of attempts on the arm by one and the total accumulated
   * value by value
   */
  Arm.prototype.reward = function(value) {

    this.count++
    this.sum += value
  }

  /**
   * @param {int} numTries
   * @param {number} totalValue
   *
   * Increments the number of attempts on the arm and the total value
   * accumulated during those attempts.
   */
  Arm.prototype.rewardMultiple = function(numTries, totalValue) {
    this.count += numTries
    this.sum += totalValue
  }

  /**
   * @returns {number} between 0 and 1
   */
  Arm.prototype.sample = function() {
    return this.rbeta(1 + this.sum, 1 + this.count - this.sum)
  }

  /**
   * @param {int} a
   * @param {int} b
   * @returns {number} - number between 0 and 1
   * 
   * The commented out code is incorrect and instead adopted the
   *jState library to sample the beta distribution.
   */
  Arm.prototype.rbeta = function(a, b) {

/**    var sum = a + b
      , ratio = a / b
      , min = Math.min(a, b)
      , lhs, rhs, y, r1, r2, lambda

    lambda = min <= 1 ?
      min :
      Math.sqrt(
        (2 * a * b - a - b) /
        (sum - 2))

    do {

      r1 = this.random()
      r2 = this.random()
      y = Math.pow(1 / r1 - 1, 1 / lambda)
      lhs = 4 * r1 * r2 * r2
      rhs =
        Math.pow(y, a - lambda) *
        Math.pow((1 + ratio) / (1 + ratio * y), sum)
    } while(lhs >= rhs)

    return ratio * y / (1 + ratio * y)
**/
    return jStat.beta.sample(a,b);
  }

//  Arm.prototype.random = Math.random;

  ///////////////////////////////////////////////////////////////////////////
  // Bandit - the n-armed bandit which selects arms from observed rewards. //
  ///////////////////////////////////////////////////////////////////////////

  /**
   * @param {object} [options]
   * @param {Array.<{count: int, sum: number}>} [options.arms]
   *      - Initialize the bandit with arms specified by the data provided
   *
   * @param {int} [options.numberOfArms]
   *      - Initialize the bandit with a number of empty arms
   *
   * Note, only one of options.arms and options.numberOfArms should
   * be provided.  If both are provided, options.arms takes precedence and
   * numberOfArms will be ignored.
   *
   * @constructor
     */
  function Bandit(options) {

    this.arms = []

    // If options.arms is explicitly passed, initialize the arms array with it
    if (options && options.arms) {
      for (var i = 0; i < options.arms.length; i++) {
        this.arms.push(new Arm(options.arms[i]))
      }
    } else { // Otherwise initialize empty arms based on options.numberOfArms
      for (var a = 0; a < (options || {}).numberOfArms; a++) {
        this.arms.push(new Arm())
      }
    }
  }

  /**
   * @returns {int} - index of the arm chosen by the bandit algorithm
   */
  Bandit.prototype.selectArm = function() {

    var max = -Infinity
      , indexOfMax = -1

    for (var armIndex = 0; armIndex < this.arms.length; armIndex++) {

      var sample = this.arms[armIndex].sample()
      if(sample > max) {

        max = sample
        indexOfMax = armIndex
      }
    }

    return indexOfMax
  }
  
  Bandit.prototype.check_convergence = function(alpha)  {
  // returns arms that can be dropped
  //
  if (this.arms.length == 1) {
    //don't do anything and return
    return [];
    }
  var success = [];
  var tries = [];
  for (var armIndex = 0; armIndex < this.arms.length; armIndex++) {
    tries.push(this.arms[armIndex].sum);
    success.push(this.arms[armIndex].count);
    }
  var failure = mathjs.subtract(tries, success);
  var ctr = mathjs.dotDivide(success, tries)
  //console.log(success);
  //console.log(tries);
  //console.log(failure);
  //console.log(ctr);
  //get index of best ctr
  var best_ctr_idx = -1;
  var best_ctr = -1;
  for (var i = 0; i < ctr.length; i++) {
    if (ctr[i] > best_ctr){
      best_ctr_idx = i
      best_ctr = ctr[i]
      }
    }
  //console.log(best_ctr)
  //console.log(best_ctr_idx)
  //now compare each are to the best arm using fisher's exact test
  var arms_to_drop = [];
  for (var i = 0; i < ctr.length; i++) {
    if (i != best_ctr_idx){
      var pvalue = this.fisher_exact([success[best_ctr_idx], success[i]],[failure[best_ctr_idx], failure[i]]);
        if(pvalue <= alpha){
          //significance!
          arms_to_drop.push(i)
          }
      }
    }
  return arms_to_drop
  }
  
  /**  
  Bandit.prototype.g_test = function(success, failure) {
  //do g_test and return p value
  // TODO: need to deal with zeros in success and failure. don't work for the g-test
  var trials = mathjs.add(success, failure);
  var expected_freq = mathjs.sum(success) / mathjs.sum(trials);
  var expected_success = mathjs.dotMultiply(trials, expected_freq);
  var expected_failure = mathjs.multiply(trials, 1 - expected_freq);
  var gsuccess = mathjs.dotMultiply(2,mathjs.sum( mathjs.dotMultiply(success,mathjs.log(mathjs.dotDivide(success, expected_success)))));
  var gfailure = mathjs.dotMultiply(2,mathjs.sum( mathjs.dotMultiply(failure,mathjs.log(mathjs.dotDivide(failure, expected_failure)))));
  var g  = gsuccess + gfailure;
  var ddof =  success.length - 1; // (number of rows - 1)(number of columns - 1) always have two rows
  //get chisq distribution value
  var pvalue = 1 - jStat.chisquare.cdf(g, ddof);
  return pvalue
  }
  **/
  
  Bandit.prototype.fisher_exact = function(success, failure) {
  //fisher's exact test https://en.wikipedia.org/wiki/Fisher%27s_exact_test
  //only works for a 2x2 contingency table
  //TODO check lengths here to make sure they're both 2
  var a = mathjs.bignumber(success[0]);
  var b = mathjs.bignumber(success[1]);
  var c = mathjs.bignumber(failure[0]);
  var d = mathjs.bignumber(failure[1]);
  var n = mathjs.add(a,b,c,d);
  var pvalue = mathjs.divide(mathjs.multiply(mathjs.combinations(mathjs.add(a,b), a),mathjs.combinations(mathjs.add(c,d), c)),mathjs.combinations(n,mathjs.add(a,c)));
  return pvalue
  }
  
  Bandit.Arm = Arm;

  exports.Bandit = Bandit

}(typeof exports === 'undefined' ? this : exports))
