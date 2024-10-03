%{
#include <stdio.h>
#include <stdlib.h>

int vars[26];  // Array to store values for variables A-Z.
int yylex(void);
void yyerror(const char *);

%}

%token VAR NUM
%token ASSIGN ADD SUB MUL DIV

%% 

input:
    /* empty */
    | input line
    ;

line:
    expr '\n'             { printf("Output -> %d\n", $1); }
    | VAR ASSIGN expr '\n' { 
        vars[$1 - 'A'] = $3; 
        printf("Output -> %d\n", $3);  // Print the evaluated RHS expression after assignment
    }
    ;

expr:
    term
    | expr ADD term { $$ = $1 + $3; }
    | expr SUB term { $$ = $1 - $3; }  // Ensure subtraction is handled properly
    ;

term:
    factor
    | term MUL factor { $$ = $1 * $3; }
    | term DIV factor {
        if ($3 == 0) {
            printf("INVALID AND THEN PROGRAM WILL EXIT.\n");
            $$ = 0; // Set the result to 0 for invalid division
        } else {
            $$ = $1 / $3;
        }
    }
    ;

factor:
    NUM
    | VAR           { $$ = vars[$1 - 'A']; }
    | '(' expr ')'  { $$ = $2; }
    ;

%% 

void yyerror(const char *s) {
    fprintf(stderr, "Error: %s\n", s);
}

int main(void) {
    printf("Simple Calculator\n");
    while (1) {
        printf("> "); // Prompt for input
        yyparse();   // Call the parser
    }
    return 0;
}
