# Collective Classification for Credibility Estimation

- collective_trust.py: Script containing Katz-based and Collective-Regression-based credibility prediction algorithms.
 See example() function for how to run the script.
- net_based_veracity_prediction.py: Flask webservice.
-- Input:
--- ego_net: Ego network of the node for which veracity should be predicted as GML string.
--- ego: Node for which veracity should be predicted.
--- type: Either truncated katz (katz) or collective regression (cr)
-- Output: Credibility score of ego

