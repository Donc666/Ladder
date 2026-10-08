# Isaac Lab Task Collection

Run rigid and articulated skills from the JSON files in `tasks/` inside Isaac Lab, and record the trajectories. Pass a task name on the command line and the program loads the JSON with that name.

## Get the code

After installing [Git](https://git-scm.com/):

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

If there is no remote repository yet, from the project directory:

```bash
git init
git add .
git commit -m "Initial public code"
```

Create an empty repository on GitHub, then:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

`.gitignore` keeps models, scenes, recorded data, and caches out of the upload. Prepare those separately.

## Environment

Use a Linux machine with an NVIDIA GPU, and install:

1. [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/). The `python` commands below use the Isaac Lab interpreter.
2. [cuRobo](https://curobo.org/). `import curobo` must succeed in that environment.
3. PartNet-Mobility meshes and URDFs. Download the official data under the PartNet license. Do not commit the full dataset to GitHub.
4. Concept-template URDFs that match the PartNet model ids.

The directories look like this:

```text
$PARTNET_MOBILITY_ROOT/<Category>/<model_id>/mobility.urdf
$PARTNET_CONCEPT_ROOT/<Category>/<model_id>/*.urdf
```

Category names match `TYPE_TO_PARTNET_CAT` in the code. Rigid objects, including the roller used in the examples below, load from `assets/` inside this repository.

By default the data lives in the repository, so the environment variables are usually unnecessary:

```text
PartNetMobility/<Category>/<model_id>/mobility.urdf
Partnet_Mobility_urdf/<Category>/<model_id>/*.urdf
assets/
```

Override the roots only when the data is stored somewhere else:

```bash
export PARTNET_MOBILITY_ROOT=/path/to/PartNetMobility
export PARTNET_CONCEPT_ROOT=/path/to/Partnet_Mobility_urdf
```

Every `python` command below uses the Isaac Lab interpreter.

## How to start

There are two ways. The first runs a task JSON that is already written. The second calls an LLM API, turns a natural-language instruction into JSON, then hands that JSON to the same scene program.

Rigid tasks live in `tasks/rigid/<task_name>.json`. Articulated tasks live in `tasks/articulated/<task_name>.json`. The `.json` suffix is optional. If the same name exists in both directories, the rigid file is used.

### 1. Run a task JSON directly

Start with `grap_roller`. It reads `tasks/rigid/grap_roller.json` and uses both arms to grab the roller on the table. The window is on and anti-interference is off:

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

Anti-interference is one data-collection switch, and it is off by default. Turning it on does three things together: after the object lifts, its mass is increased to see whether the grasp slips; while the object pose changes during motion, the end effector follows that pose; and the run checks whether the grasp actually lifted the object, retrying when it did not. Turning it off skips all three. The end effector keeps the pose recorded at import, the trajectory plays through the waypoints, and grasp success is not judged.

- On: `--anti_interference`
- Off: `--no_anti_interference` (default)

```bash
python run_task.py --task grap_roller --enable_cameras --anti_interference
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

With a window:

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

`task_agent.py` accepts `--anti_interference` and `--no_anti_interference` and forwards the flag to `run_task.py`. The default is off.

To turn anti-interference on for the same roller instruction:

```bash
python task_agent.py --instruction "use both arms to grab the roller on the table" --provider deepseek --anti_interference
```

To save the generated JSON somewhere else:

```bash
python task_agent.py --instruction "use both arms to grab the roller on the table" --provider openai --output ./tasks/rigid/grap_roller.json --no_anti_interference
```

Model names and base URLs have defaults. Override them only when needed:

```bash
export DEEPSEEK_MODEL="deepseek-chat"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="gpt-4o"
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

## License

Add a `LICENSE` file before publishing the repository. PartNet-Mobility, Lightwheel scenes, and assets shipped with Isaac each have their own license. Open-sourcing this code does not make those models public.
