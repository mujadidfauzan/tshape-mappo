"""Research extensions: variable teams, task diagnostics, curriculum and ablations."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from pettingzoo.test import parallel_api_test

from src.envs.config import EnvironmentConfig, load_config
from src.envs.diagnostics import EpisodeDiagnostics
from src.envs.rewards import team_reward
from src.envs.termination import footprint_error, footprint_margin
from src.envs.tshape_env import TShapeParallelEnv
from src.training.curriculum import Curriculum
from src.training.experiments import experiment_jobs


class ResearchEnvironmentTests(unittest.TestCase):
    def test_all_agent_counts_reset_step_and_parallel_api_in_all_arenas(self):
        for arena in ("environment.yaml", "environment_medium.yaml", "environment_hard.yaml"):
            for n in (2, 4, 8, 16):
                with self.subTest(arena=arena, num_agents=n):
                    env = TShapeParallelEnv(config=replace(load_config(Path("configs") / arena), num_agents=n, max_steps=3))
                    try:
                        self.assertEqual(env.physics.agent_names, [f"agent_{i}" for i in range(n)])
                        self.assertEqual(env.obs_dim, 18+4*n)
                        self.assertEqual(env.state_space.shape, (17+4*n,))
                        self.assertEqual(env.physics.model.nu, 2*n)
                        for seed in range(8):
                            obs, info = env.reset(seed=seed)
                            self.assertEqual(len(obs), n)
                            self.assertTrue(env.physics.spawn_is_valid(.04))
                            pos, _ = env.physics.get_agents_state()
                            gaps = env.physics.object_surface_distances(pos) - env.physics.radii
                            self.assertTrue(np.all(gaps <= .15 + 1e-10))
                        parallel_api_test(env, num_cycles=5)
                    finally:
                        env.close()

    def test_short_goal_distance_and_yaw_constraints_are_actually_sampled(self):
        config = load_config()
        curriculum = Curriculum.load("configs/curricula/easy.yaml")
        curriculum.validate_configs(config)
        env = TShapeParallelEnv(config=curriculum.config(config))
        try:
            for seed in range(100):
                env.reset(seed=seed)
                self.assertGreaterEqual(env._distance_to_goal(), .8-1e-10)
                self.assertLessEqual(env._distance_to_goal(), 1.2+1e-10)
                self.assertLessEqual(abs(env.physics.get_object_state()["yaw"]), .15)
                self.assertFalse(env.object_inside_goal())
        finally:
            env.close()

    def test_diagnostics_distinguish_center_crossing_full_crossing_and_settling(self):
        env = TShapeParallelEnv(config="configs/environment_medium.yaml")
        try:
            env.reset(seed=42)
            p = env.physics
            p.reset([.2, 0, np.pi/2])
            d = EpisodeDiagnostics(p, env.goal_center, env.goal_half_size)
            d.update(p, 1, False, False)
            self.assertEqual(d.report()["gates_center_crossed"], [True])
            self.assertEqual(d.report()["gates_fully_cleared"], [False])
            self.assertEqual(d.report(terminal=True)["failure_category"], "gate_blocked")
            p.reset([.8, 0, np.pi/2]); d.update(p, 2, False, False)
            self.assertEqual(d.report()["gates_fully_cleared"], [True])
            p.reset([1.8, 1.5, 0]);d.update(p, 3, True, False, np.full(4, .05))
            report = d.report(terminal=True)
            self.assertEqual(report["failure_category"], "goal_not_settled")
            self.assertEqual(report["inside_but_unsettled_steps"], 1)
            np.testing.assert_allclose(report["agent_object_contact_time_s"], .05)
            self.assertEqual(d.report(success=True, terminal=True)["failure_category"], "success")
        finally:
            env.close()

    def test_footprint_error_detects_protrusion_at_zero_com_distance(self):
        env = TShapeParallelEnv()
        try:
            p = env.physics
            p.reset([0, 0, np.arctan(.5)])
            corners = p.object_corners()
            self.assertGreater(footprint_error(corners, [0,0], [.75,.65]), .02)
            self.assertLess(footprint_margin(corners, [0,0], [.75,.65]), -.02)
            p.reset([0,0,0]);self.assertEqual(footprint_error(p.object_corners(), [0,0], [.75,.65]), 0)
        finally:
            env.close()

    def test_potential_reward_telescopes_and_timeout_does_not_zero_the_potential(self):
        cfg = EnvironmentConfig(reward_mode="footprint_potential", shaping_gamma=.9, progress_weight=1,
                                success_reward=0, time_penalty=0, failure_penalty=0)
        errors = [2., 1., .5]
        r0 = team_reward(4,3,False,False,cfg,errors[0],errors[1])[0]
        r1 = team_reward(3,2,False,False,cfg,errors[1],errors[2])[0]
        self.assertAlmostEqual(r0 + .9*r1, errors[0] - .9**2*errors[2])
        terminal = team_reward(3,2,True,False,cfg,1.,.5)[0]
        self.assertAlmostEqual(terminal, 1.)
        failure = team_reward(3,2,False,True,cfg,1.,.5)[0]
        self.assertAlmostEqual(failure, 1.)

    def test_curriculum_validation_promotion_streak_and_restore(self):
        spec = {"difficulty":"easy", "validation_interval_steps":2, "validation_episodes":2,
                "minimum_stage_steps":2, "consecutive_checks":2,
                "stages":[{"name":"start", "max_steps":2}, {"name":"full"}]}
        c = Curriculum(spec);c.validate_configs(EnvironmentConfig())
        self.assertFalse(c.observe(.8,2));self.assertFalse(c.observe(.1,4))
        self.assertFalse(c.observe(.9,6));self.assertTrue(c.observe(.9,8))
        restored = Curriculum(spec,c.state_dict())
        self.assertEqual(restored.name,"full")
        self.assertFalse(restored.observe(.9,10));self.assertFalse(restored.observe(.9,12))
        self.assertTrue(restored.completed)
        self.assertFalse(restored.due(14))
        with self.assertRaises(ValueError):
            Curriculum({"stages":[{"name":"wrong", "mass":9}]})

    def test_experiment_matrix_has_independent_policies_and_equal_team_budgets(self):
        _, jobs = experiment_jobs("configs/experiments/agent_count.yaml")
        self.assertEqual(len(jobs),20)
        self.assertEqual({e.num_agents for _,_,e in jobs},{2,4,8,16})
        self.assertEqual({c.total_steps for _,c,_ in jobs},{2000000})
        self.assertEqual(len({name for name,_,_ in jobs}),20)
        with self.assertRaises(ValueError):
            experiment_jobs({"num_agents":[4,4]})

    def test_every_curriculum_stage_supports_small_and_large_teams(self):
        for task in ("easy", "medium", "hard"):
            path = "configs/environment.yaml" if task == "easy" else f"configs/environment_{task}.yaml"
            curriculum = Curriculum.load(f"configs/curricula/{task}.yaml")
            for n in (2, 16):
                base = replace(load_config(path), num_agents=n)
                curriculum.validate_configs(base)
                for stage in range(3):
                    env = TShapeParallelEnv(config=curriculum.config(base, stage))
                    try:
                        for seed in range(3):
                            env.reset(seed=seed)
                            self.assertTrue(env.physics.spawn_is_valid(.04))
                    finally:
                        env.close()

    def test_group_summary_separates_best_latest_and_reports_seed_variation(self):
        from src.evaluation.metrics import aggregate_benchmarks
        common = {"task_id":"a", "protocol_id":"b", "num_agents":4, "difficulty":"easy",
                  "reward_mode":"baseline", "mode":"deterministic", "checkpoint_kind":"best",
                  "stable_rate":1., "mean_return":2., "mean_completion_time_s":None}
        rows = [{**common,"training_seed":11,"success_rate":.2},
                {**common,"training_seed":22,"success_rate":.6},
                {**common,"checkpoint_kind":"latest","training_seed":11,"success_rate":0.}]
        groups = aggregate_benchmarks(rows)
        best = next(g for g in groups if g["checkpoint_kind"]=="best")
        self.assertEqual(best["unique_training_seeds"],2)
        self.assertAlmostEqual(best["success_rate_mean"],.4)
        self.assertAlmostEqual(best["success_rate_std"],np.std([.2,.6],ddof=1))
        self.assertIsNone(next(g for g in groups if g["checkpoint_kind"]=="latest")["success_rate_std"])


@unittest.skipIf(importlib.util.find_spec("torch") is None,
                 "install requirements-training.txt for research training tests")
class ResearchTrainingTests(unittest.TestCase):
    def config(self, **values):
        from src.training.config import MAPPOConfig
        return MAPPOConfig(total_steps=4,rollout_steps=2,ppo_epochs=1,actor_hidden_size=8,critic_hidden_size=8,
                           eval_episodes=1,eval_interval_steps=2,video_interval_steps=0,tensorboard=False,**values)

    def test_variable_agent_checkpoint_reload_is_self_contained(self):
        from src.algorithms.mappo import MAPPO
        from src.evaluation.checkpoints import loaded_policy
        from src.training.trainer import Trainer
        for n in (2,8,16):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);trainer=Trainer(root/'run',self.config(),{"num_agents":n,"max_steps":2})
                trainer.train()
                checkpoint=root/'run/checkpoints/latest.pt'
                _,payload=MAPPO.from_checkpoint(checkpoint)
                self.assertEqual(payload['dimensions']['num_agents'],n)
                self.assertEqual(payload['training']['model_xml'].count('childclass="agent"'),n)
                with loaded_policy(checkpoint) as (agent,env,_):
                    self.assertEqual(agent.dimensions['observation_size'],18+4*n)
                    self.assertTrue(env.resolved_model_path.exists())
                resumed=Trainer(root/'resumed',self.config(),resume=checkpoint,resume_step_budget=6)
                self.assertEqual(resumed.env.config.num_agents,n)
                resumed.train()

    def test_curriculum_transition_checkpoint_and_resume_do_not_depend_on_original_yaml(self):
        from src.algorithms.mappo import MAPPO
        from src.training.trainer import Trainer
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'curriculum.yaml'
            spec={"difficulty":"easy","validation_interval_steps":2,"validation_episodes":1,
                  "minimum_stage_steps":2,"consecutive_checks":1,
                  "stages":[{"name":"short","max_steps":8,"object_goal_distance_range":[.8,1.2]}, {"name":"full"}]}
            path.write_text(yaml.safe_dump(spec))
            trainer=Trainer(root/'run',self.config(curriculum=str(path)),{"num_agents":2,"max_steps":2})
            score={"reports":[],"summary":{"success_rate":1.,"mean_return":20.}}
            # Controlled validation scores test the state machine, not learning.
            with patch('src.training.trainer.evaluate_policy',return_value=score.copy()):
                trainer.train()
            _,payload=MAPPO.from_checkpoint(root/'run/checkpoints/latest.pt')
            self.assertTrue(payload['training']['curriculum_state']['completed'])
            transition=json.loads((root/'run/curriculum_transitions.jsonl').read_text().splitlines()[0])
            self.assertEqual(transition['discarded_partial_episode_steps'],2)
            self.assertEqual(payload['training']['completed_episodes'],1)
            path.unlink()
            resumed=Trainer(root/'resumed',self.config(),resume=root/'run/checkpoints/latest.pt',resume_step_budget=6)
            self.assertTrue(resumed.curriculum.completed)
            self.assertEqual(resumed.curriculum.name,'full')
            resumed.train()

    def test_actor_transfer_preserves_actor_function_but_creates_fresh_critic_and_optimizers(self):
        import torch
        from src.algorithms.mappo import MAPPO
        from src.training.trainer import Trainer
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);Trainer(root/'source',self.config(),{'max_steps':2}).train()
            checkpoint=root/'source/checkpoints/latest.pt';source,_=MAPPO.from_checkpoint(checkpoint)
            target=Trainer(root/'target',replace(self.config(),seed=43),
                replace(load_config('configs/environment_medium.yaml'),max_steps=2),init_actor=checkpoint)
            try:
                inputs=np.random.default_rng(4).normal(size=(4,34))
                np.testing.assert_array_equal(source.act(inputs,True)[0],target.agent.act(inputs,True)[0])
                self.assertEqual(len(target.agent.actor_optimizer.state),0)
                self.assertEqual(target.agent.state_normalizer.count.item(),0)
                self.assertTrue(any(not torch.equal(a,b) for a,b in zip(source.critic.parameters(),target.agent.critic.parameters())))
            finally:
                target.env.close();target.logger.close()
            with self.assertRaisesRegex(ValueError,'same agent count'):
                Trainer(root/'bad',self.config(),{'max_steps':2,'num_agents':2},init_actor=checkpoint)

    def test_benchmark_and_experiment_runner_cli_write_compact_results(self):
        from src.training.trainer import Trainer
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);Trainer(root/'source',self.config(),{'max_steps':2,'num_agents':2}).train()
            cmd=[sys.executable,'scripts/benchmark.py',str(root/'source/checkpoints/best.pt'),
                 str(root/'source/checkpoints/latest.pt'),'--episodes','2','--device','cpu','--output',str(root/'benchmark')]
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            lines=(root/'benchmark/episodes.jsonl').read_text().splitlines()
            self.assertEqual(len(lines),8)
            self.assertEqual(len((root/'benchmark/summary.csv').read_text().splitlines()),5)
            self.assertTrue(json.loads((root/'benchmark/manifest.json').read_text())['complete'])
            env=root/'env.yaml';env.write_text(yaml.safe_dump({'max_steps':2}))
            algo=root/'mappo.yaml';algo.write_text(yaml.safe_dump(self.config().to_dict()))
            matrix=root/'matrix.yaml';matrix.write_text(yaml.safe_dump({'environment_config':str(env),
                'mappo_config':str(algo),'num_agents':[2,8],'seeds':[11]}))
            result=subprocess.run([sys.executable,'scripts/run_experiments.py',str(matrix),'--device','cpu',
                '--output',str(root/'experiments')],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            rows=(root/'experiments/experiments.csv').read_text()
            self.assertEqual(rows.count('complete'),2)

    def test_footprint_potential_runs_ppo_and_rejects_a_different_discount(self):
        from src.training.trainer import Trainer
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,"must equal MAPPO gamma"):
                Trainer(root/'bad',self.config(),{'reward_mode':'footprint_potential','shaping_gamma':.9})
            trainer=Trainer(root/'run',self.config(),{'reward_mode':'footprint_potential','num_agents':8,'max_steps':2})
            summary=trainer.train()
            self.assertEqual(summary['environment_steps'],4)
            records=[json.loads(l) for l in (root/'run/updates.jsonl').read_text().splitlines()]
            self.assertTrue(all(np.isfinite(r['value_loss']) for r in records))

    def test_training_seeds_skip_curriculum_and_heldout_ranges(self):
        from src.training.trainer import Trainer
        trainer=Trainer.__new__(Trainer)
        trainer.config=replace(self.config(),seed=149995)
        trainer.curriculum=Curriculum({'stages':[{'name':'full'}]})
        for index in range(50):
            trainer.episode_index=index
            self.assertFalse(150000 <= trainer.episode_seed() < 150020)
        trainer.config=replace(self.config(),seed=199998)
        seeds=[]
        for index in range(4):
            trainer.episode_index=index;seeds.append(trainer.episode_seed())
        self.assertEqual(seeds,[199998,199999,300000,300001])


if __name__ == '__main__':
    unittest.main()
