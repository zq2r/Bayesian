import unittest

import torch

from eval.process_reward import posterior_rewards, resolve_conservatism


class PosteriorRewardsTest(unittest.TestCase):
    def test_conservatism_reweights_toward_lower_reward_heads(self):
        mu = torch.tensor([[0.2, 0.8], [0.1, 0.4]])
        logits = torch.zeros_like(mu)

        plain = posterior_rewards(mu, logits, False, 0.1)
        conservative = posterior_rewards(mu, logits, True, 0.1)

        torch.testing.assert_close(plain["mu_final"], plain["mu_rel"])
        self.assertTrue(torch.all(conservative["mu_final"] < plain["mu_final"]))
        self.assertTrue(torch.all(conservative["post_weights"][:, 0] > 0.5))
        torch.testing.assert_close(
            conservative["post_weights"].sum(dim=-1), torch.ones(2)
        )

    def test_checkpoint_defaults_and_override(self):
        class Model:
            belief_use_conservatism = True
            belief_conservatism_beta = 0.25

        self.assertEqual(resolve_conservatism(Model(), "auto", None), (True, 0.25))
        self.assertEqual(resolve_conservatism(Model(), "false", 0.5), (False, 0.5))
        with self.assertRaises(ValueError):
            resolve_conservatism(Model(), "true", 0)


if __name__ == "__main__":
    unittest.main()
