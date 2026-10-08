# Isaac Lab Task Collection

Run rigid and articulated skills from the JSON files in `tasks/` inside Isaac Lab, and record the trajectories. Pass a task name on the command line and the program loads the JSON with that name.

## Get the code


```bash
git clone https://github.com/Donc666/Ladder.git
cd Ladder
```

## Get the models and scenes

Install the Hugging Face download tool, then from the `Ladder` directory pull the dataset into the repository root. The folders land next to `run_task.py`:

```bash
pip install -U huggingface_hub
hf download sdasfdsafg5asdf/Ladder-assets --repo-type dataset --local-dir .
```

That creates:

```text
assets/
PartNetMobility/<Category>/<model_id>/mobility.urdf
Partnet_Mobility_urdf/<Category>/<model_id>/*.urdf
Lightwheel_Kitchen/
Lightwheel_Bedroom/
Lightwheel_Toyroom/
robots/
Grasp_Editor_Tutorial_Stage/
```



Rigid objects, including the roller used below, load from `assets/`. Articulated objects load from `PartNetMobility/`. Category names match `TYPE_TO_PARTNET_CAT`. These files stay out of git. Do not commit them.

After the download above, do not set any path. The program looks for `PartNetMobility/` and `Partnet_Mobility_urdf/` next to `run_task.py`.

Set these two variables only when those folders are stored somewhere else. Replace the placeholder with the real directory:

```bash
export PARTNET_MOBILITY_ROOT=/path/to/PartNetMobility
export PARTNET_CONCEPT_ROOT=/path/to/Partnet_Mobility_urdf
```

## Environment

Use a Linux machine with an NVIDIA GPU, and install:

1. [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) 5.1.0-rc.19 and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 0.54.3. The installed Isaac Sim build is `5.1.0-rc.19+release.26219.9c81211b.gl`. The `python` commands below use that environment's interpreter (Python 3.11).
2. [cuRobo](https://curobo.org/) `nvidia-curobo` 0.8.0.post1.dev35, from commit `ec2bfa9f9ea2673abe00cf911ef39353ddb79c29` on the cuRobo `main` branch. `import curobo` must succeed in that environment.
3. The model and scene folders from the Hugging Face dataset above.

Every `python` command below uses the Isaac Lab interpreter.

## How to start

There are two ways. The first runs a task JSON that is already written. The second calls an LLM API, turns a natural-language instruction into JSON, then hands that JSON to the same scene program.

Rigid tasks live in `tasks/rigid/<task_name>.json`. Articulated tasks live in `tasks/articulated/<task_name>.json`. The `.json` suffix is optional. If the same name exists in both directories, the rigid file is used.

### 1. Run a task JSON directly

Start with `grap_roller`. It reads `tasks/rigid/grap_roller.json` and uses both arms to grab the roller on the table. This command opens the Isaac Lab simulation window, so the robot and the scene are visible while the task runs. Anti-interference stays off. Anti-interference is an extra feature in the LADDER pipeline. In simulation it imitates external disturbances from the real world, such as an object accidentally dropping while it is being grasped. Turning it on lets the pipeline resist that disturbance and grasp the object again. The feature is only applied to some tasks right now, for example `stack_bowls_three`. `grap_roller` does not retry after a drop.

```bash
python run_task.py --task grap_roller --enable_cameras --no_anti_interference
```

Add `--headless` when you do not want a window:

```bash
python run_task.py --task grap_roller --enable_cameras --headless --no_anti_interference
```

`--num_envs` sets how many environments run in parallel. The default is 1:

```bash
python run_task.py --task grap_roller --enable_cameras --no_anti_interference --num_envs 4
python run_task.py --task grap_roller --enable_cameras --headless --no_anti_interference --num_envs 4
```

- On: `--anti_interference`
- Off: `--no_anti_interference` (default)

```bash
python run_task.py --task stack_bowls_three --enable_cameras --anti_interference
```

Other tasks use the same command with `--task` set to a file name under `tasks/rigid/` or `tasks/articulated/`. Omitting `--task` loads `tasks/rigid/task_plan.json`.

Collect many episodes:

```bash
bash ./collect.sh 25 --task grap_roller --enable_cameras --headless --no_anti_interference
```

Multiple GPUs:

```bash
./run_microwave_scene_copy_parallel.sh 25 0,1 --task grap_roller --no_anti_interference
```

### 2. Generate a task with an LLM, then run it

`task_agent.py` sends one natural-language instruction to an LLM, validates the JSON, saves it, and launches `run_task.py` on that file.

Install the API dependencies first (this does not install Isaac Lab):

```bash
pip install openai pydantic
```

Keep keys in environment variables. Do not write them into `task_agent.py`, and do not commit them. Set the variable for the provider you use, in the current terminal:

```bash
# DeepSeek, the default provider
export DEEPSEEK_API_KEY="your-key"

# OpenAI
export OPENAI_API_KEY="your-key"

# Moonshot / Kimi
export KIMI_API_KEY="your-key"
```

Create a key in the provider console: DeepSeek [API keys](https://platform.deepseek.com/api_keys), OpenAI [API keys](https://platform.openai.com/api-keys), Kimi [Moonshot console](https://platform.moonshot.cn/console/api-keys).

`--provider` is `deepseek` (default), `openai`, or `kimi`. The generated JSON is written to `tasks/rigid/task_plan.json`, and the scene program then runs that file. The examples below ask for the roller grasp.

With the Isaac Lab simulation window:

```bash
python task_agent.py --instruction "use both arms to grab the roller on the table" --provider deepseek --no_anti_interference
```

Without a window:

```bash
python task_agent.py --instruction "use both arms to grab the roller on the table" --provider deepseek --headless --no_anti_interference
```

`--num_envs` sets the environment count here as well. The default is 1:

```bash
python task_agent.py --instruction "use both arms to grab the roller on the table" --provider deepseek --no_anti_interference --num_envs 4
python task_agent.py --instruction "use both arms to grab the roller on the table" --provider deepseek --headless --no_anti_interference --num_envs 4
```

`task_agent.py` accepts `--anti_interference` and `--no_anti_interference` and forwards the flag to `run_task.py`. The default is off. Anti-interference is the same LADDER feature described above: it imitates a real-world disturbance, such as a drop during a grasp, and grasps again. It currently applies only to some tasks, for example `stack_bowls_three`.

To turn anti-interference on for that task:

```bash
python task_agent.py --instruction "stack three bowls" --provider deepseek --anti_interference
```

To save the generated JSON somewhere else:

```bash
python task_agent.py --instruction "use both arms to grab the roller on the table" --provider openai --output ./tasks/rigid/grap_roller.json --no_anti_interference
```

Model names and base URLs have defaults. Override them only when needed:

```bash
export DEEPSEEK_MODEL="deepseek-chat"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="gpt-5.6"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export KIMI_MODEL="kimi-k3"
export KIMI_BASE_URL="https://api.moonshot.cn/v1"
```

If the key for the chosen provider is missing, the program exits and names the environment variable that is unset.

## Task JSON

```json
{
  "task_name": "grap_roller",
  "assets": [
    {"id": "roller_1", "type": "roller", "asset_kind": "auto"},
    {"id": "table_1", "type": "table", "asset_kind": "auto"}
  ],
  "atomic_plan": [
    {"asset": "roller_1", "atom": "grap_simulataneously", "params": {}}
  ]
}
```

Asset parameters and atomic skills live in `manipulation knowledge base/`. Each rigid category is one file. Each articulated category is one file too: `SPEC` holds placement and grasp parameters, and the class methods are the atomic skills. `fraction` overrides the default open or close amount.


