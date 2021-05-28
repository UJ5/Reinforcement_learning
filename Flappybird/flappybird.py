import pygame,sys, random


def draw_floor():
	screen.blit(floor_surface,(floor_x_position,450))
	screen.blit(floor_surface,(floor_x_position+288,450))

def create_pipe():
	random_pipe_pos=  random.choice(pipe_height)
	bottom_pipe = pipe_surface.get_rect(midtop=(350,random_pipe_pos))
	top_pipe = pipe_surface.get_rect(midbottom=(350,random_pipe_pos-120))
	return bottom_pipe, top_pipe
 
def move_pipes(pipes):
	for pipe in pipes:
		pipe.centerx -= 3
	return pipes

def draw_pipes(pipes):
	for pipe in pipes:
		if pipe.bottom>=512 :
			screen.blit(pipe_surface,pipe)
		else:
			flip_pipe=pygame.transform.flip(pipe_surface,False,True)
			screen.blit(flip_pipe,pipe)

def check_collision(pipes):
	for pipe in pipes:
		if bird_rect.colliderect(pipe):
			return False
	if bird_rect.top <= -100 or bird_rect.bottom >= 800:
		return False
	return True


def rotate_bird(bird):
	new_bird = pygame.transform.rotozoom(bird, bird_movement*-3,1)
	return new_bird

def bird_animation():
	new_bird = bird_frames[bird_index]
	new_bird_rect = new_bird.get_rect(center=(50,bird_rect.centery))
	return new_bird, new_bird_rect

def score_display(game_state):
	if game_state == 'main_game':
		score_surface = game_font.render(f'Score:{int(score)}',True,(255,255,255))
		score_rect = score_surface.get_rect(center=(144, 30))
		screen.blit(score_surface,score_rect)
	if game_state == 'game_over':
		score_surface = game_font.render(f'Score:{int(score)}',True,(255,255,255))
		score_rect = score_surface.get_rect(center=(144, 30))
		screen.blit(score_surface,score_rect)

		highscore_surface = game_font.render(f'Highscore:{int(high_score)}',True,(255,255,255))
		highscore_rect = highscore_surface.get_rect(center=(144, 400 ))
		screen.blit(highscore_surface,highscore_rect)

		
def update_score(score,high_score):
	if score > high_score:
		high_score = score
	return high_score


pygame.init()

screen = pygame.display.set_mode((288,512))
clock = pygame.time.Clock()
game_font = pygame.font.Font('FlappyBird_Python-master/04B_19.TTF',20)

#game_variables
gravity=0.25
bird_movement = 0
game_active = True
score = 0
high_score = 0


bg_surface= pygame.image.load("FlappyBird_Python-master/assets/background-day.png").convert()
floor_surface= pygame.image.load("FlappyBird_Python-master/assets/base.png").convert()
floor_x_position = 0

bird_downflap = pygame.image.load("FlappyBird_Python-master/assets/bluebird-downflap.png").convert_alpha()
bird_midflap = pygame.image.load("FlappyBird_Python-master/assets/bluebird-midflap.png").convert_alpha()
bird_upflap = pygame.image.load("FlappyBird_Python-master/assets/bluebird-upflap.png").convert_alpha()
bird_frames = [bird_upflap,bird_midflap,bird_downflap]
bird_index = 0
bird_surface = bird_frames[bird_index]
bird_rect = bird_surface.get_rect(center = (50,256))

BIRDFLAP = pygame.USEREVENT+1
pygame.time.set_timer(BIRDFLAP,200)

pipe_surface = pygame.image.load("FlappyBird_Python-master/assets/pipe-green.png")
pipe_list = []
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer (SPAWNPIPE,1200)
pipe_height = [200,300,350]

game_over_surface = pygame.image.load("FlappyBird_Python-master/assets/message.png").convert_alpha()
game_over_rect = game_over_surface.get_rect(center=(144,256))

while True:
	for event in pygame.event.get():
		if (event.type == pygame.QUIT):
			pygame.quit()
			sys.exit()
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_SPACE:
				bird_movement = 0 
				bird_movement -= 5
			if event.key == pygame.K_SPACE and game_active==False:
				game_active = True
				pipe_list.clear()
				bird_rect.center=(50,256)
				bird_movement = 0
				score = 0

		if event.type == SPAWNPIPE:
			pipe_list.extend(create_pipe())	

		if event.type == BIRDFLAP:
			if bird_index <2:
				bird_index +=1
			else :   
				bird_index = 0
			bird_surface, bird_rect = bird_animation()
			
	screen.blit(bg_surface,(0,0))
	if game_active == True:
		bird_movement += gravity
		rotated_bird = rotate_bird(bird_surface)
		bird_rect.centery += bird_movement
		screen.blit(rotated_bird,bird_rect)
		game_active=check_collision(pipe_list)
		pipe_list = move_pipes(pipe_list)
		draw_pipes(pipe_list)
		score += 0.01
		score_display('main_game')
	else:
		screen.blit(game_over_surface,game_over_rect)
		high_score = update_score(score,high_score)
		score_display('game_over')



	floor_x_position -=1
	draw_floor()
	if (floor_x_position <= -288):
		floor_x_position=0		
	pygame.display.update()
	clock.tick(120)
